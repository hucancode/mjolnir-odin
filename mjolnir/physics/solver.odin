package physics

import "base:intrinsics"
import "core:math"
import "core:math/bits"
import "core:math/linalg"
import "core:mem"
import "core:sync"
import "core:thread"

// TGS-soft contact solver
// Collide once per frame; each substep: integrate velocities → warm start →
// solve with soft bias → integrate positions → relax (solve without bias).
// A restitution pass runs once after all substeps. Soft constraints replace
// Baumgarte/pseudo-velocity position correction: the bias is a critically
// tuned spring (contact hertz/damping) whose energy the relax pass removes.
SLOP :: 0.002
RESTITUTION_THRESHOLD :: 1.0 // m/s approach speed below which no bounce
CONTACT_HERTZ :: 30.0
CONTACT_DAMPING_RATIO :: 10.0
// Static contacts are stiffer (2x hertz, half damping) to resist push-through
STATIC_HERTZ_SCALE :: 2.0
STATIC_DAMPING_SCALE :: 0.5
CONTACT_MAX_PUSH_SPEED :: 3.0 // m/s cap on penetration recovery
SOLVER_PARALLEL_THRESHOLD :: 200

Softness :: struct {
  bias_rate:     f32,
  mass_scale:    f32,
  impulse_scale: f32,
}

make_soft :: proc "contextless" (hertz, zeta, h: f32) -> Softness {
  if hertz <= 0 do return {0, 1, 0}
  omega := 2.0 * math.PI * hertz
  a1 := 2.0 * zeta + h * omega
  a2 := h * omega * a1
  a3 := 1.0 / (1.0 + a2)
  return {omega / a1, a2 * a3, a3}
}

SpinBarrier :: struct {
  arrive: i32,
  sense:  bool,
}

@(private)
spin_barrier_wait :: proc "contextless" (b: ^SpinBarrier, local_sense: ^bool, expected: i32) {
  local_sense^ = !local_sense^
  cur := sync.atomic_add(&b.arrive, 1) + 1
  if cur == expected {
    sync.atomic_store(&b.arrive, 0)
    sync.atomic_store(&b.sense, local_sense^)
    return
  }
  for sync.atomic_load(&b.sense) != local_sense^ {
    intrinsics.cpu_relax()
  }
}

// ---------------------------------------------------------------------------
// Graph coloring partition (parallel solve without locks)
// ---------------------------------------------------------------------------

build_solver_partition :: proc(world: ^World, num_shards: int) {
  body_count := len(world.bodies.entries)
  resize(&world.solver_color_used, body_count)
  if body_count > 0 {
    mem.zero_slice(world.solver_color_used[:])
  }
  for &b in world.solver_color_buckets do clear(&b)
  clear(&world.solver_overflow)
  world.solver_color_count = 0
  for c, idx in world.dynamic_contacts {
    a_idx := int(c.body_a.index)
    b_idx := int(c.body_b.index)
    if a_idx >= body_count || b_idx >= body_count do continue
    used := world.solver_color_used[a_idx] | world.solver_color_used[b_idx]
    free_mask := ~used
    if free_mask == 0 {
      // No color left for this body pair: solve single-threaded instead of
      // racing on color 0.
      append(&world.solver_overflow, idx)
      continue
    }
    color := int(bits.trailing_zeros(free_mask))
    bit := u64(1) << u64(color)
    world.solver_color_used[a_idx] |= bit
    world.solver_color_used[b_idx] |= bit
    if color >= len(world.solver_color_buckets) {
      old_len := len(world.solver_color_buckets)
      resize(&world.solver_color_buckets, color + 1)
      for i in old_len ..< color + 1 {
        world.solver_color_buckets[i] = make([dynamic]int)
      }
    }
    append(&world.solver_color_buckets[color], idx)
    if color + 1 > world.solver_color_count {
      world.solver_color_count = color + 1
    }
  }
  shard_count := max(1, num_shards)
  if shard_count != world.solver_static_shard_count {
    for &s in world.solver_static_shards do delete(s)
    delete(world.solver_static_shards)
    world.solver_static_shards = make([][dynamic]int, shard_count)
    for i in 0 ..< shard_count {
      world.solver_static_shards[i] = make([dynamic]int)
    }
    world.solver_static_shard_count = shard_count
  } else {
    for i in 0 ..< shard_count do clear(&world.solver_static_shards[i])
  }
  for c, idx in world.static_contacts {
    s := int(c.body_a.index) % shard_count
    append(&world.solver_static_shards[s], idx)
  }
}

// ---------------------------------------------------------------------------
// Prepare
// ---------------------------------------------------------------------------

// Inverse of the coupled 2x2 tangent effective-mass matrix at the manifold
// centroid. Coupling the two tangent directions keeps the friction cone
// circular — independent per-axis clamps inflate μ by up to √2 on diagonals.
@(private = "file")
invert_tangent_k :: #force_inline proc(k11, k12, k22: f32) -> matrix[2, 2]f32 {
  det := k11 * k22 - k12 * k12
  if math.abs(det) < math.F32_EPSILON do return {}
  inv_det := 1.0 / det
  return matrix[2, 2]f32{
    k22 * inv_det, -k12 * inv_det,
    -k12 * inv_det, k11 * inv_det,
  }
}

// Generic over the body-B type: B == DynamicRigidBody compiles in body B's
// mass/inertia/velocity contributions; B == StaticRigidBody compiles them out
// (a static body is a zero-inv-mass body).
prepare_contact :: proc(
  contact: ^$C,
  body_a: ^DynamicRigidBody,
  body_b: ^$B,
  dt: f32,
) {
  inv_mass_sum := body_a.inv_mass
  when B == DynamicRigidBody do inv_mass_sum += body_b.inv_mass
  centroid: [3]f32
  for i in 0 ..< contact.count {
    p := &contact.points[i]
    centroid += p.point
    p.r_a = p.point - body_a.position
    r_a_cross_n := linalg.cross(p.r_a, contact.normal)
    normal_mass := inv_mass_sum + linalg.dot(body_a.inv_inertia_world * r_a_cross_n, r_a_cross_n)
    rel := -(body_a.velocity + linalg.cross(body_a.angular_velocity, p.r_a))
    when B == DynamicRigidBody {
      p.r_b = p.point - body_b.position
      r_b_cross_n := linalg.cross(p.r_b, contact.normal)
      normal_mass += linalg.dot(body_b.inv_inertia_world * r_b_cross_n, r_b_cross_n)
      rel += body_b.velocity + linalg.cross(body_b.angular_velocity, p.r_b)
    }
    p.normal_mass = normal_mass > math.F32_EPSILON ? 1.0 / normal_mass : 0
    p.base_separation = -p.penetration
    p.relative_velocity = linalg.dot(rel, contact.normal)
    p.max_normal_impulse = 0
  }
  centroid /= f32(contact.count)
  contact.r_a_c = centroid - body_a.position
  contact.tangent1, contact.tangent2 = compute_tangent_basis(contact.normal)
  rt_a1 := linalg.cross(contact.r_a_c, contact.tangent1)
  rt_a2 := linalg.cross(contact.r_a_c, contact.tangent2)
  ia1 := body_a.inv_inertia_world * rt_a1
  ia2 := body_a.inv_inertia_world * rt_a2
  k11 := inv_mass_sum + linalg.dot(ia1, rt_a1)
  k12 := linalg.dot(ia1, rt_a2)
  k22 := inv_mass_sum + linalg.dot(ia2, rt_a2)
  kt := linalg.dot(contact.normal, body_a.inv_inertia_world * contact.normal)
  when B == DynamicRigidBody {
    contact.r_b_c = centroid - body_b.position
    rt_b1 := linalg.cross(contact.r_b_c, contact.tangent1)
    rt_b2 := linalg.cross(contact.r_b_c, contact.tangent2)
    ib1 := body_b.inv_inertia_world * rt_b1
    ib2 := body_b.inv_inertia_world * rt_b2
    k11 += linalg.dot(ib1, rt_b1)
    k12 += linalg.dot(ib1, rt_b2)
    k22 += linalg.dot(ib2, rt_b2)
    kt += linalg.dot(contact.normal, body_b.inv_inertia_world * contact.normal)
  }
  contact.tangent_mass = invert_tangent_k(k11, k12, k22)
  contact.twist_mass = kt > math.F32_EPSILON ? 1.0 / kt : 0
}

// ---------------------------------------------------------------------------
// Warm start (per substep)
// ---------------------------------------------------------------------------

warmstart_contact :: proc(
  contact: ^$C,
  body_a: ^DynamicRigidBody,
  body_b: ^$B,
) {
  for i in 0 ..< contact.count {
    p := &contact.points[i]
    impulse_n := contact.normal * p.normal_impulse
    apply_impulse_at_point_no_wake(body_a, -impulse_n, p.point)
    when B == DynamicRigidBody do apply_impulse_at_point_no_wake(body_b, impulse_n, p.point)
  }
  centroid := body_a.position + contact.r_a_c
  impulse_t := contact.tangent1 * contact.tangent_impulse[0] +
    contact.tangent2 * contact.tangent_impulse[1]
  apply_impulse_at_point_no_wake(body_a, -impulse_t, centroid)
  when B == DynamicRigidBody do apply_impulse_at_point_no_wake(body_b, impulse_t, centroid)
  twist := contact.normal * contact.twist_impulse
  if body_a.enable_rotation {
    body_a.angular_velocity -= body_a.inv_inertia_world * twist
  }
  when B == DynamicRigidBody {
    if body_b.enable_rotation {
      body_b.angular_velocity += body_b.inv_inertia_world * twist
    }
  }
}

// ---------------------------------------------------------------------------
// Velocity solve (bias and relax passes share this)
// ---------------------------------------------------------------------------

// Current separation: base + how much the anchors moved along the normal
// since prepare (translation-only approximation of box3d's delta tracking).
@(private = "file")
current_separation :: #force_inline proc(
  p: ^ContactPoint,
  normal: [3]f32,
  body_a: ^DynamicRigidBody,
  body_b: ^$B,
) -> f32 {
  ds := -(body_a.position - body_a.position0)
  when B == DynamicRigidBody do ds += body_b.position - body_b.position0
  return p.base_separation + linalg.dot(ds, normal)
}

resolve_velocity :: proc(
  contact: ^$C,
  body_a: ^DynamicRigidBody,
  body_b: ^$B,
  soft: Softness,
  inv_h: f32,
  use_bias: bool,
) {
  total_normal_impulse: f32 = 0
  lever_sum: f32 = 0
  centroid := body_a.position + contact.r_a_c
  for i in 0 ..< contact.count {
    p := &contact.points[i]
    s := current_separation(p, contact.normal, body_a, body_b)
    bias: f32 = 0
    mass_scale: f32 = 1
    impulse_scale: f32 = 0
    if s > 0 {
      // Speculative: only stop the approach, never pull
      bias = s * inv_h
    } else if use_bias {
      bias = max(soft.bias_rate * s, -CONTACT_MAX_PUSH_SPEED)
      mass_scale = soft.mass_scale
      impulse_scale = soft.impulse_scale
    }
    rel_p := -(body_a.velocity + linalg.cross(body_a.angular_velocity, p.r_a))
    when B == DynamicRigidBody {
      rel_p += body_b.velocity + linalg.cross(body_b.angular_velocity, p.r_b)
    }
    vn := linalg.dot(rel_p, contact.normal)
    delta_impulse := -p.normal_mass * mass_scale * (vn + bias) - impulse_scale * p.normal_impulse
    old_impulse := p.normal_impulse
    p.normal_impulse = max(old_impulse + delta_impulse, 0.0)
    p.max_normal_impulse = max(p.max_normal_impulse, p.normal_impulse)
    impulse := contact.normal * (p.normal_impulse - old_impulse)
    apply_impulse_at_point_no_wake(body_a, -impulse, p.point)
    when B == DynamicRigidBody do apply_impulse_at_point_no_wake(body_b, impulse, p.point)
    total_normal_impulse += p.normal_impulse
    lever_sum += p.normal_impulse * linalg.length(p.point - centroid)
  }
  // Coupled tangent friction at the manifold centroid, circular cone clamp
  rel := -(body_a.velocity + linalg.cross(body_a.angular_velocity, contact.r_a_c))
  when B == DynamicRigidBody {
    rel += body_b.velocity + linalg.cross(body_b.angular_velocity, contact.r_b_c)
  }
  vt := [2]f32{linalg.dot(rel, contact.tangent1), linalg.dot(rel, contact.tangent2)}
  delta_t := contact.tangent_mass * -vt
  old_t := contact.tangent_impulse
  new_t := old_t + delta_t
  max_friction := contact.friction * total_normal_impulse
  len_t := linalg.length(new_t)
  if len_t > max_friction {
    new_t *= max_friction / max(len_t, math.F32_EPSILON)
  }
  contact.tangent_impulse = new_t
  applied_t := new_t - old_t
  impulse_t := contact.tangent1 * applied_t[0] + contact.tangent2 * applied_t[1]
  apply_impulse_at_point_no_wake(body_a, -impulse_t, centroid)
  when B == DynamicRigidBody do apply_impulse_at_point_no_wake(body_b, impulse_t, centroid)
  // Twist friction about the normal — resists spinning even on flat rest.
  // Torque budget comes from the normal impulses' lever arms.
  wt := linalg.dot(-body_a.angular_velocity, contact.normal)
  when B == DynamicRigidBody {
    wt += linalg.dot(body_b.angular_velocity, contact.normal)
  }
  delta_w := contact.twist_mass * -wt
  max_twist := contact.friction * lever_sum
  old_w := contact.twist_impulse
  contact.twist_impulse = clamp(old_w + delta_w, -max_twist, max_twist)
  twist := contact.normal * (contact.twist_impulse - old_w)
  if body_a.enable_rotation {
    body_a.angular_velocity -= body_a.inv_inertia_world * twist
  }
  when B == DynamicRigidBody {
    if body_b.enable_rotation {
      body_b.angular_velocity += body_b.inv_inertia_world * twist
    }
  }
}

// ---------------------------------------------------------------------------
// Restitution pass (once, after all substeps)
// ---------------------------------------------------------------------------

// Applied separately so the soft bias never doubles as bounce, and gated on
// max_normal_impulse so speculative points that never touched don't bounce.
apply_restitution :: proc(
  contact: ^$C,
  body_a: ^DynamicRigidBody,
  body_b: ^$B,
) {
  if contact.restitution == 0 do return
  for i in 0 ..< contact.count {
    p := &contact.points[i]
    if p.relative_velocity > -RESTITUTION_THRESHOLD || p.max_normal_impulse == 0 do continue
    rel := -(body_a.velocity + linalg.cross(body_a.angular_velocity, p.r_a))
    when B == DynamicRigidBody {
      rel += body_b.velocity + linalg.cross(body_b.angular_velocity, p.r_b)
    }
    vn := linalg.dot(rel, contact.normal)
    delta_impulse := -p.normal_mass * (vn + contact.restitution * p.relative_velocity)
    old_impulse := p.normal_impulse
    p.normal_impulse = max(old_impulse + delta_impulse, 0.0)
    impulse := contact.normal * (p.normal_impulse - old_impulse)
    apply_impulse_at_point_no_wake(body_a, -impulse, p.point)
    when B == DynamicRigidBody do apply_impulse_at_point_no_wake(body_b, impulse, p.point)
  }
}

// ---------------------------------------------------------------------------
// Substep loop — sequential
// ---------------------------------------------------------------------------

@(private = "file")
SolverPass :: enum {
  Warmstart,
  Bias,
  Relax,
  Restitution,
}

// One pass over a contact slice. indices == nil means the whole slice.
// Generic over contact type — handle types on the contact pick the right pools.
@(private = "file")
run_pass :: proc(
  world: ^World,
  contacts: []$C,
  indices: []int,
  pass: SolverPass,
  soft: Softness,
  inv_h: f32,
) {
  n := indices != nil ? len(indices) : len(contacts)
  #no_bounds_check for k in 0 ..< n {
    c := &contacts[indices != nil ? indices[k] : k]
    a := get(world, c.body_a) or_continue
    b := get(world, c.body_b) or_continue
    switch pass {
    case .Warmstart:
      warmstart_contact(c, a, b)
    case .Bias:
      resolve_velocity(c, a, b, soft, inv_h, true)
    case .Relax:
      resolve_velocity(c, a, b, soft, inv_h, false)
    case .Restitution:
      apply_restitution(c, a, b)
    }
  }
}

@(private = "file")
sequential_pass :: proc(world: ^World, pass: SolverPass, soft, static_soft: Softness, inv_h: f32) {
  run_pass(world, world.dynamic_contacts[:], nil, pass, soft, inv_h)
  run_pass(world, world.static_contacts[:], nil, pass, static_soft, inv_h)
}

run_substep_loop_sequential :: proc(world: ^World, h: f32, num_substeps: int, ccd_handled: []bool) {
  inv_h := h > 0 ? 1.0 / h : 0
  soft := make_soft(min(CONTACT_HERTZ, 0.125 * inv_h), CONTACT_DAMPING_RATIO, h)
  static_soft := make_soft(
    min(STATIC_HERTZ_SCALE * CONTACT_HERTZ, 0.25 * inv_h),
    STATIC_DAMPING_SCALE * CONTACT_DAMPING_RATIO,
    h,
  )
  for _ in 0 ..< num_substeps {
    integrate_velocities_range(world, 0, len(world.awake_list), h)
    sequential_pass(world, .Warmstart, soft, static_soft, inv_h)
    sequential_pass(world, .Bias, soft, static_soft, inv_h)
    integrate_positions_range(world, 0, len(world.awake_list), h, ccd_handled)
    sequential_pass(world, .Relax, soft, static_soft, inv_h)
  }
  sequential_pass(world, .Restitution, soft, static_soft, inv_h)
}

// ---------------------------------------------------------------------------
// Substep loop — parallel (graph colors + static shards + overflow)
// ---------------------------------------------------------------------------

SolverWorkerData :: struct {
  world:        ^World,
  thread_id:    int,
  num_threads:  int,
  num_substeps: int,
  h:            f32,
  ccd_handled:  []bool,
  barrier:      ^SpinBarrier,
}

@(private = "file")
worker_contact_phase :: proc(
  data: ^SolverWorkerData,
  pass: SolverPass,
  soft, static_soft: Softness,
  inv_h: f32,
  local_sense: ^bool,
  expected: i32,
) {
  world := data.world
  for color_idx in 0 ..< world.solver_color_count {
    bucket := world.solver_color_buckets[color_idx][:]
    bucket_len := len(bucket)
    if bucket_len > 0 {
      chunk := (bucket_len + data.num_threads - 1) / data.num_threads
      s := data.thread_id * chunk
      e := min(s + chunk, bucket_len)
      if s < e {
        run_pass(world, world.dynamic_contacts[:], bucket[s:e], pass, soft, inv_h)
      }
    }
    spin_barrier_wait(data.barrier, local_sense, expected)
  }
  // Overflow contacts share bodies with every color; single thread only.
  if data.thread_id == 0 {
    run_pass(world, world.dynamic_contacts[:], world.solver_overflow[:], pass, soft, inv_h)
  }
  spin_barrier_wait(data.barrier, local_sense, expected)
  if data.thread_id < world.solver_static_shard_count {
    shard := world.solver_static_shards[data.thread_id][:]
    run_pass(world, world.static_contacts[:], shard, pass, static_soft, inv_h)
  }
  spin_barrier_wait(data.barrier, local_sense, expected)
}

solver_worker_task :: proc(task: thread.Task) {
  data := (^SolverWorkerData)(task.data)
  world := data.world
  expected := i32(data.num_threads)
  local_sense := false
  inv_h := data.h > 0 ? 1.0 / data.h : 0
  soft := make_soft(min(CONTACT_HERTZ, 0.125 * inv_h), CONTACT_DAMPING_RATIO, data.h)
  static_soft := make_soft(
    min(STATIC_HERTZ_SCALE * CONTACT_HERTZ, 0.25 * inv_h),
    STATIC_DAMPING_SCALE * CONTACT_DAMPING_RATIO,
    data.h,
  )
  body_count := len(world.awake_list)
  chunk := (body_count + data.num_threads - 1) / data.num_threads
  b_start := min(data.thread_id * chunk, body_count)
  b_end := min(b_start + chunk, body_count)
  for _ in 0 ..< data.num_substeps {
    integrate_velocities_range(world, b_start, b_end, data.h)
    spin_barrier_wait(data.barrier, &local_sense, expected)
    worker_contact_phase(data, .Warmstart, soft, static_soft, inv_h, &local_sense, expected)
    worker_contact_phase(data, .Bias, soft, static_soft, inv_h, &local_sense, expected)
    integrate_positions_range(world, b_start, b_end, data.h, data.ccd_handled)
    spin_barrier_wait(data.barrier, &local_sense, expected)
    worker_contact_phase(data, .Relax, soft, static_soft, inv_h, &local_sense, expected)
  }
  worker_contact_phase(data, .Restitution, soft, static_soft, inv_h, &local_sense, expected)
}

run_substep_loop :: proc(world: ^World, h: f32, num_substeps: int, ccd_handled: []bool, num_threads: int) {
  total_contacts := len(world.dynamic_contacts) + len(world.static_contacts)
  if total_contacts < SOLVER_PARALLEL_THRESHOLD || num_threads <= 1 || !world.enable_parallel {
    run_substep_loop_sequential(world, h, num_substeps, ccd_handled)
    return
  }
  build_solver_partition(world, num_threads)
  barrier := SpinBarrier{}
  task_data := make([]SolverWorkerData, num_threads, context.temp_allocator)
  for t in 0 ..< num_threads {
    task_data[t] = SolverWorkerData {
      world        = world,
      thread_id    = t,
      num_threads  = num_threads,
      num_substeps = num_substeps,
      h            = h,
      ccd_handled  = ccd_handled,
      barrier      = &barrier,
    }
    thread.pool_add_task(&world.thread_pool, mem.nil_allocator(), solver_worker_task, &task_data[t], t)
  }
  pool_wait(&world.thread_pool)
}

compute_tangent_basis :: proc(normal: [3]f32) -> ([3]f32, [3]f32) {
  tangent1 := linalg.orthogonal(normal)
  tangent2 := linalg.cross(normal, tangent1)
  return tangent1, tangent2
}
