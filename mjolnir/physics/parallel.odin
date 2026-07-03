package physics

import cont "../containers"
import "../geometry"
import "base:intrinsics"
import "core:log"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:sync"
import "core:thread"
import "core:time"

DEFAULT_THREAD_COUNT :: 16

pool_wait :: proc(pool: ^thread.Pool) {
  for {
    task, ok := thread.pool_pop_waiting(pool)
    if !ok do break
    thread.pool_do_work(pool, task)
  }
  for thread.pool_num_outstanding(pool) > 0 {
    intrinsics.cpu_relax()
  }
}

BVH_Refit_Task_Data :: struct {
  physics: ^World,
  start:   int,
  end:     int,
}

CCD_Work_Queue :: struct {
  current_index: i32,
  total_count:   int,
}

CCD_Task_Data_Dynamic :: struct {
  physics:          ^World,
  work_queue:       ^CCD_Work_Queue,
  dt:               f32,
  ccd_handled:      []bool,
  bodies_tested:    int,
  total_candidates: int,
}

Prepare_Task_Data :: struct {
  physics: ^World,
  start:   int,
  end:     int,
  dt:      f32,
}

prepare_dynamic_task :: proc(task: thread.Task) {
  data := (^Prepare_Task_Data)(task.data)
  contacts := data.physics.dynamic_contacts[:]
  #no_bounds_check for i in data.start ..< data.end {
    c := &contacts[i]
    body_a := get(data.physics, c.body_a) or_continue
    body_b := get(data.physics, c.body_b) or_continue
    prepare_contact(c, body_a, body_b, data.dt)
  }
}

prepare_static_task :: proc(task: thread.Task) {
  data := (^Prepare_Task_Data)(task.data)
  contacts := data.physics.static_contacts[:]
  #no_bounds_check for i in data.start ..< data.end {
    c := &contacts[i]
    body_a := get(data.physics, c.body_a) or_continue
    body_b := get(data.physics, c.body_b) or_continue
    prepare_contact(c, body_a, body_b, data.dt)
  }
}

sequential_prepare_contacts :: proc(world: ^World, dt: f32) {
  for &c in world.dynamic_contacts {
    a := get(world, c.body_a) or_continue
    b := get(world, c.body_b) or_continue
    prepare_contact(&c, a, b, dt)
  }
  for &c in world.static_contacts {
    a := get(world, c.body_a) or_continue
    b := get(world, c.body_b) or_continue
    prepare_contact(&c, a, b, dt)
  }
}

parallel_prepare_contacts :: proc(world: ^World, dt: f32, num_threads := DEFAULT_THREAD_COUNT) {
  dyn_count := len(world.dynamic_contacts)
  sta_count := len(world.static_contacts)
  total := dyn_count + sta_count
  if total < 200 || num_threads <= 1 {
    sequential_prepare_contacts(world, dt)
    return
  }
  task_data := make([]Prepare_Task_Data, num_threads * 2, context.temp_allocator)
  task_idx := 0
  if dyn_count > 0 {
    chunk := (dyn_count + num_threads - 1) / num_threads
    for i in 0 ..< num_threads {
      s := i * chunk
      if s >= dyn_count do break
      e := min(s + chunk, dyn_count)
      task_data[task_idx] = Prepare_Task_Data{physics = world, start = s, end = e, dt = dt}
      thread.pool_add_task(&world.thread_pool, mem.nil_allocator(), prepare_dynamic_task, &task_data[task_idx], task_idx)
      task_idx += 1
    }
  }
  if sta_count > 0 {
    chunk := (sta_count + num_threads - 1) / num_threads
    for i in 0 ..< num_threads {
      s := i * chunk
      if s >= sta_count do break
      e := min(s + chunk, sta_count)
      task_data[task_idx] = Prepare_Task_Data{physics = world, start = s, end = e, dt = dt}
      thread.pool_add_task(&world.thread_pool, mem.nil_allocator(), prepare_static_task, &task_data[task_idx], task_idx)
      task_idx += 1
    }
  }
  pool_wait(&world.thread_pool)
}

refit_range :: #force_inline proc(physics: ^World, start, end: int) {
  #no_bounds_check for i in start ..< end {
    bvh_entry := &physics.dynamic_bvh.primitives[i]
    body := get(physics, bvh_entry.handle) or_continue
    if body.is_killed || body.is_sleeping do continue
    bvh_entry.bounds = geometry.Aabb {
      min = body.cached_aabb.min - SPECULATIVE_DISTANCE,
      max = body.cached_aabb.max + SPECULATIVE_DISTANCE,
    }
  }
}

bvh_refit_task :: proc(task: thread.Task) {
  data := (^BVH_Refit_Task_Data)(task.data)
  refit_range(data.physics, data.start, data.end)
}

parallel_bvh_refit :: proc(
  physics: ^World,
  num_threads := DEFAULT_THREAD_COUNT,
) {
  primitive_count := len(physics.dynamic_bvh.primitives)
  if primitive_count == 0 do return
  if primitive_count < 100 || num_threads == 1 {
    sequential_bvh_refit(physics)
    return
  }
  chunk_size := (primitive_count + num_threads - 1) / num_threads
  task_data_array := make(
    []BVH_Refit_Task_Data,
    num_threads,
    context.temp_allocator,
  )
  for i in 0 ..< num_threads {
    start := i * chunk_size
    end := min(start + chunk_size, primitive_count)
    if start >= primitive_count do break
    task_data_array[i] = BVH_Refit_Task_Data {
      physics = physics,
      start   = start,
      end     = end,
    }
    thread.pool_add_task(
      &physics.thread_pool,
      mem.nil_allocator(),
      bvh_refit_task,
      &task_data_array[i],
      i,
    )
  }
  pool_wait(&physics.thread_pool)
  geometry.bvh_refit(&physics.dynamic_bvh)
}

sequential_bvh_refit :: proc(physics: ^World) {
  refit_range(physics, 0, len(physics.dynamic_bvh.primitives))
  geometry.bvh_refit(&physics.dynamic_bvh)
}

// Narrowphase one broadphase pair into a contact. Generic over the B-side
// handle (dynamic or static) and matching contact type. Returns 1 if the
// narrowphase test actually ran (pre-culls return 0).
narrowphase_pair :: #force_inline proc(
  physics: ^World,
  handle_a: DynamicRigidBodyHandle,
  handle_b: $H,
  warmstart: ^map[u64]ContactWarmstart,
  out: ^[dynamic]$C,
) -> (narrow_tests: int) {
  body_a := get(physics, handle_a) or_else nil
  body_b := get(physics, handle_b) or_else nil
  if body_a == nil || body_b == nil do return
  if body_a.is_killed do return
  when H == DynamicRigidBodyHandle {
    if body_b.is_killed do return
    if body_a.is_sleeping && body_b.is_sleeping do return
  } else {
    if body_a.is_sleeping do return
  }
  if !bounding_spheres_intersect(
    body_a.cached_sphere_center, body_a.cached_sphere_radius + SPECULATIVE_DISTANCE,
    body_b.cached_sphere_center, body_b.cached_sphere_radius,
  ) {
    return
  }
  narrow_tests = 1
  manifold, hit := collide_bodies(body_a, body_b, SPECULATIVE_DISTANCE)
  if !hit do return
  if body_a.is_sleeping do wake_up(body_a)
  when H == DynamicRigidBodyHandle {
    if body_b.is_sleeping do wake_up(body_b)
  }
  contact := C {
    body_a      = handle_a,
    body_b      = handle_b,
    normal      = manifold.normal,
    count       = manifold.count,
    restitution = mix_restitution(body_a.restitution, body_b.restitution),
    friction    = mix_friction(body_a.friction, body_b.friction),
  }
  for i in 0 ..< manifold.count {
    contact.points[i] = ContactPoint {
      point       = manifold.points[i].point,
      penetration = manifold.points[i].penetration,
      feature_id  = manifold.points[i].feature_id,
    }
  }
  if w, found := &warmstart[collision_pair_hash(handle_a, handle_b)]; found {
    contact_apply_warmstart(&contact, w)
  }
  append(out, contact)
  return
}

broadphase_collect_pairs :: proc(physics: ^World) -> (
  dynamic_pairs: [dynamic]geometry.BVHOverlapPair(DynamicBroadPhaseEntry),
  static_pairs: [dynamic]geometry.BVHCrossPair(DynamicBroadPhaseEntry, StaticBroadPhaseEntry),
) {
  dynamic_pairs = make([dynamic]geometry.BVHOverlapPair(DynamicBroadPhaseEntry), context.temp_allocator)
  static_pairs = make([dynamic]geometry.BVHCrossPair(DynamicBroadPhaseEntry, StaticBroadPhaseEntry), context.temp_allocator)
  geometry.bvh_find_all_overlaps(&physics.dynamic_bvh, &dynamic_pairs)
  geometry.bvh_find_cross_overlaps(&physics.dynamic_bvh, &physics.static_bvh, &static_pairs)
  return
}

@(private = "file")
narrowphase_all :: proc(
  physics: ^World,
  dynamic_pairs: []geometry.BVHOverlapPair(DynamicBroadPhaseEntry),
  static_pairs: []geometry.BVHCrossPair(DynamicBroadPhaseEntry, StaticBroadPhaseEntry),
) {
  for pair in dynamic_pairs {
    narrowphase_pair(physics, pair.a.handle, pair.b.handle, &physics.prev_dynamic_warmstart, &physics.dynamic_contacts)
  }
  for pair in static_pairs {
    narrowphase_pair(physics, pair.a.handle, pair.b.handle, &physics.prev_static_warmstart, &physics.static_contacts)
  }
}

sequential_collision_detection_traversal :: proc(physics: ^World) {
  dynamic_pairs, static_pairs := broadphase_collect_pairs(physics)
  narrowphase_all(physics, dynamic_pairs[:], static_pairs[:])
}

ccd_step_body :: proc(
  physics: ^World,
  body_a: ^DynamicRigidBody,
  idx_a: int,
  dt: f32,
  ccd_handled: []bool,
  static_candidates: ^[dynamic]StaticBroadPhaseEntry,
) -> (tested: bool, candidate_count: int) {
  if body_a.is_killed || body_a.is_sleeping do return
  collider_a := &body_a.collider
  motion := body_a.velocity * dt
  motion_len_sq := linalg.length2(motion)
  min_extent := collider_min_extent(collider_a)
  threshold := 0.5 * min_extent
  if motion_len_sq < threshold * threshold do return
  tested = true
  earliest_toi := f32(1.0)
  has_ccd_hit := false
  swept_aabb := geometry.Aabb {
    min = linalg.min(body_a.cached_aabb.min, body_a.cached_aabb.min + motion),
    max = linalg.max(body_a.cached_aabb.max, body_a.cached_aabb.max + motion),
  }
  clear(static_candidates)
  geometry.bvh_query_aabb(&physics.static_bvh, swept_aabb, static_candidates)
  candidate_count = len(static_candidates)
  for candidate in static_candidates {
    body_b := get(physics, candidate.handle) or_continue
    toi := swept_test(collider_a, &body_b.collider, body_a.position, body_b.position, body_a.rotation, body_b.rotation, motion)
    if toi.has_impact && toi.time < earliest_toi {
      earliest_toi = toi.time
      has_ccd_hit = true
    }
  }
  if !(has_ccd_hit && earliest_toi > 0.01 && earliest_toi < 0.99) do return
  // Park just short of the impact; narrowphase runs after CCD this frame and
  // the speculative contact stops the remaining approach.
  body_a.position += motion * (earliest_toi * 0.98)
  update_cached_aabb(&body_a.base)
  wake_up(body_a)
  ccd_handled[idx_a] = true
  return
}

ccd_task_dynamic :: proc(task: thread.Task) {
  data := (^CCD_Task_Data_Dynamic)(task.data)
  static_candidates := make([dynamic]StaticBroadPhaseEntry, 0, 64, context.temp_allocator)
  BATCH_SIZE :: 32
  for {
    start_idx := int(sync.atomic_add(&data.work_queue.current_index, i32(BATCH_SIZE)))
    if start_idx >= data.work_queue.total_count do break
    end_idx := min(start_idx + BATCH_SIZE, data.work_queue.total_count)
    #no_bounds_check for idx_a in start_idx ..< end_idx {
      if idx_a >= len(data.physics.bodies.entries) do break
      entry_a := &data.physics.bodies.entries[idx_a]
      if !entry_a.active do continue
      tested, cands := ccd_step_body(data.physics, &entry_a.item, idx_a, data.dt, data.ccd_handled, &static_candidates)
      if tested do data.bodies_tested += 1
      data.total_candidates += cands
    }
  }
}

parallel_ccd :: proc(
  physics: ^World,
  dt: f32,
  ccd_handled: []bool,
  num_threads := DEFAULT_THREAD_COUNT,
) -> (
  bodies_tested: int,
  total_candidates: int,
) {
  body_count := len(physics.bodies.entries)
  if body_count == 0 do return
  if body_count < 100 || num_threads == 1 {
    return sequential_ccd(physics, dt, ccd_handled)
  }
  work_queue := CCD_Work_Queue {
    current_index = 0,
    total_count   = body_count,
  }
  task_data_array := make(
    []CCD_Task_Data_Dynamic,
    num_threads,
    context.temp_allocator,
  )
  for i in 0 ..< num_threads {
    task_data_array[i] = CCD_Task_Data_Dynamic {
      physics     = physics,
      work_queue  = &work_queue,
      dt          = dt,
      ccd_handled = ccd_handled,
    }
    thread.pool_add_task(
      &physics.thread_pool,
      mem.nil_allocator(),
      ccd_task_dynamic,
      &task_data_array[i],
      i,
    )
  }
  pool_wait(&physics.thread_pool)
  for &task_data in task_data_array {
    bodies_tested += task_data.bodies_tested
    total_candidates += task_data.total_candidates
  }
  return
}

sequential_ccd :: proc(
  physics: ^World,
  dt: f32,
  ccd_handled: []bool,
) -> (bodies_tested: int, total_candidates: int) {
  static_candidates := make([dynamic]StaticBroadPhaseEntry, 0, 64, context.temp_allocator)
  #no_bounds_check for &entry_a, idx_a in physics.bodies.entries do if entry_a.active {
    tested, cands := ccd_step_body(physics, &entry_a.item, idx_a, dt, ccd_handled, &static_candidates)
    if tested do bodies_tested += 1
    total_candidates += cands
  }
  return
}

// Collision detection using BVH tree-vs-tree traversal: O(N + K) where K is
// the number of overlapping pairs. Pairs are split across threads; each thread
// appends into its own contact arrays which are concatenated afterwards.
Collision_Detection_Task_Data_Traversal :: struct {
  physics:          ^World,
  dynamic_pairs:    []geometry.BVHOverlapPair(DynamicBroadPhaseEntry),
  static_pairs:     []geometry.BVHCrossPair(DynamicBroadPhaseEntry, StaticBroadPhaseEntry),
  start:            int,
  end:              int,
  dynamic_contacts: [dynamic]DynamicContact,
  static_contacts:  [dynamic]StaticContact,
}

collision_detection_task_traversal :: proc(task: thread.Task) {
  data := (^Collision_Detection_Task_Data_Traversal)(task.data)
  // Dynamic pairs in [start, end) clipped to dynamic_pairs range
  dyn_end := min(data.end, len(data.dynamic_pairs))
  #no_bounds_check for i in data.start ..< dyn_end {
    pair := data.dynamic_pairs[i]
    narrowphase_pair(data.physics, pair.a.handle, pair.b.handle, &data.physics.prev_dynamic_warmstart, &data.dynamic_contacts)
  }
  static_start := max(0, data.start - len(data.dynamic_pairs))
  static_end := min(data.end - len(data.dynamic_pairs), len(data.static_pairs))
  #no_bounds_check for i in static_start ..< static_end {
    pair := data.static_pairs[i]
    narrowphase_pair(data.physics, pair.a.handle, pair.b.handle, &data.physics.prev_static_warmstart, &data.static_contacts)
  }
}

parallel_collision_detection_traversal :: proc(
  self: ^World,
  num_threads := DEFAULT_THREAD_COUNT,
) {
  if len(self.dynamic_bvh.primitives) == 0 do return
  dynamic_pairs, static_pairs := broadphase_collect_pairs(self)
  total_pairs := len(dynamic_pairs) + len(static_pairs)
  when ENABLE_VERBOSE_LOG {
    log.infof(
      "Tree traversal found %d dynamic pairs + %d static pairs",
      len(dynamic_pairs),
      len(static_pairs),
    )
  }
  if total_pairs == 0 do return
  if total_pairs < 100 || num_threads == 1 {
    narrowphase_all(self, dynamic_pairs[:], static_pairs[:])
    return
  }

  per_thread_dyn_cap := max(64, len(dynamic_pairs) / max(1, num_threads) + 32)
  per_thread_sta_cap := max(64, len(static_pairs) / max(1, num_threads) + 32)
  pairs_per_thread := (total_pairs + num_threads - 1) / num_threads
  task_data_array := make(
    []Collision_Detection_Task_Data_Traversal,
    num_threads,
    context.temp_allocator,
  )
  for i in 0 ..< num_threads {
    start := i * pairs_per_thread
    end := min((i + 1) * pairs_per_thread, total_pairs)
    if start >= total_pairs do break
    task_data_array[i] = Collision_Detection_Task_Data_Traversal {
      physics          = self,
      dynamic_pairs    = dynamic_pairs[:],
      static_pairs     = static_pairs[:],
      start            = start,
      end              = end,
      dynamic_contacts = make([dynamic]DynamicContact, 0, per_thread_dyn_cap, context.temp_allocator),
      static_contacts  = make([dynamic]StaticContact, 0, per_thread_sta_cap, context.temp_allocator),
    }
    thread.pool_add_task(
      &self.thread_pool,
      context.allocator,
      collision_detection_task_traversal,
      &task_data_array[i],
      i,
    )
  }
  pool_wait(&self.thread_pool)
  for &task_data in task_data_array {
    for contact in task_data.dynamic_contacts {
      append(&self.dynamic_contacts, contact)
    }
    for contact in task_data.static_contacts {
      append(&self.static_contacts, contact)
    }
  }
}
