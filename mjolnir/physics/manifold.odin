package physics

// Multi-point contact manifold generation. Analytic only — primitives are
// sphere, box, cylinder (fan is trigger-only). Multi-point manifolds are what
// let boxes rest flat and stacks stand still: a single averaged point cannot
// resist tipping or twisting.

import "../geometry"
import "core:math"
import "core:math/linalg"

MAX_MANIFOLD_POINTS :: 4

ManifoldPoint :: struct {
  point:       [3]f32,
  penetration: f32,
  feature_id:  u32,
}

Manifold :: struct {
  normal: [3]f32, // from A to B
  count:  int,
  points: [MAX_MANIFOLD_POINTS]ManifoldPoint,
}

@(private = "file")
FEATURE_EDGE_CONTACT :: u32(0xE0000000)
@(private = "file")
FEATURE_SINGLE :: u32(0xF0000000)

manifold_from_single_point :: #force_inline proc(
  point: [3]f32,
  normal: [3]f32,
  penetration: f32,
) -> (m: Manifold) {
  m.normal = normal
  m.count = 1
  m.points[0] = {point = point, penetration = penetration, feature_id = FEATURE_SINGLE}
  return
}

// ---------------------------------------------------------------------------
// Box-box
// ---------------------------------------------------------------------------

@(private = "file")
BoxFrame :: struct {
  center: [3]f32,
  axes:   [3][3]f32, // world-space unit axes
  half:   [3]f32,
}

@(private = "file")
box_frame :: proc(pos: [3]f32, rot: quaternion128, half: [3]f32) -> (f: BoxFrame) {
  r := linalg.matrix3_from_quaternion(rot)
  f.center = pos
  f.axes = {
    {r[0, 0], r[1, 0], r[2, 0]},
    {r[0, 1], r[1, 1], r[2, 1]},
    {r[0, 2], r[1, 2], r[2, 2]},
  }
  f.half = half
  return
}

// Support extent of a box along a world direction
@(private = "file")
box_extent :: #force_inline proc(f: ^BoxFrame, dir: [3]f32) -> f32 {
  return(
    f.half.x * math.abs(linalg.dot(dir, f.axes[0])) +
    f.half.y * math.abs(linalg.dot(dir, f.axes[1])) +
    f.half.z * math.abs(linalg.dot(dir, f.axes[2])) \
  )
}

// World-space corners of face `axis_idx` (sign = ±1) of a box, CCW seen from
// outside. Corner order is deterministic — feature ids depend on it.
@(private = "file")
box_face_vertices :: proc(f: ^BoxFrame, axis_idx: int, sign: f32) -> (verts: [4][3]f32) {
  u_idx := (axis_idx + 1) % 3
  v_idx := (axis_idx + 2) % 3
  n := f.axes[axis_idx] * (sign * f.half[axis_idx])
  u := f.axes[u_idx] * f.half[u_idx]
  v := f.axes[v_idx] * f.half[v_idx]
  c := f.center + n
  verts[0] = c - u - v
  verts[1] = c + u - v
  verts[2] = c + u + v
  verts[3] = c - u + v
  return
}

@(private = "file")
ClipVertex :: struct {
  p:  [3]f32,
  id: u32,
}

// Sutherland-Hodgman clip of a polygon against plane dot(n,x) <= d.
// New vertices get ids derived from the edge they were created on plus the
// clip plane index, so they stay stable frame-to-frame.
@(private = "file")
clip_polygon_plane :: proc(
  poly: []ClipVertex,
  n: [3]f32,
  d: f32,
  plane_id: u32,
  out: ^[8]ClipVertex,
) -> (count: int) {
  n_in := len(poly)
  if n_in == 0 do return 0
  prev := poly[n_in - 1]
  prev_dist := linalg.dot(n, prev.p) - d
  for i in 0 ..< n_in {
    cur := poly[i]
    cur_dist := linalg.dot(n, cur.p) - d
    if prev_dist <= 0 && count < 8 {
      out[count] = prev
      count += 1
    }
    if (prev_dist < 0) != (cur_dist < 0) && count < 8 {
      t := prev_dist / (prev_dist - cur_dist)
      out[count] = {
        p  = prev.p + (cur.p - prev.p) * t,
        id = (prev.id ~ cur.id) << 8 | plane_id | 0x40000000,
      }
      count += 1
    }
    prev = cur
    prev_dist = cur_dist
  }
  return
}

// Reduce n candidate points to <=4 keeping the deepest point and maximal
@(private = "file")
reduce_manifold :: proc(m: ^Manifold, candidates: []ManifoldPoint) {
  n := len(candidates)
  if n <= MAX_MANIFOLD_POINTS {
    for c, i in candidates do m.points[i] = c
    m.count = n
    return
  }
  // 1: deepest
  best := 0
  for i in 1 ..< n {
    if candidates[i].penetration > candidates[best].penetration do best = i
  }
  m.points[0] = candidates[best]
  // 2: farthest from 1
  best2, best_d := 0, f32(-1)
  for c, i in candidates {
    d := linalg.length2(c.point - m.points[0].point)
    if d > best_d {
      best_d = d
      best2 = i
    }
  }
  m.points[1] = candidates[best2]
  // 3: max triangle area
  best3, best_a := 0, f32(-1)
  e01 := m.points[1].point - m.points[0].point
  for c, i in candidates {
    a := linalg.length2(linalg.cross(e01, c.point - m.points[0].point))
    if a > best_a {
      best_a = a
      best3 = i
    }
  }
  m.points[2] = candidates[best3]
  // 4: farthest from the triangle plane-projected centroid (max spread)
  centroid := (m.points[0].point + m.points[1].point + m.points[2].point) / 3
  best4, best_d4 := 0, f32(-1)
  for c, i in candidates {
    d := linalg.length2(c.point - centroid)
    if d > best_d4 {
      best_d4 = d
      best4 = i
    }
  }
  m.points[3] = candidates[best4]
  m.count = 4
}

// Axis-aligned fast path: overlap box, face manifold on the min axis.
@(private = "file")
collide_boxes_aligned :: proc(
  pos_a: [3]f32,
  half_a: [3]f32,
  pos_b: [3]f32,
  half_b: [3]f32,
  margin: f32,
) -> (m: Manifold, hit: bool) {
  min_a := pos_a - half_a
  max_a := pos_a + half_a
  min_b := pos_b - half_b
  max_b := pos_b + half_b
  overlap_min := linalg.max(min_a, min_b)
  overlap_max := linalg.min(max_a, max_b)
  overlap := overlap_max - overlap_min
  // Speculative: accept up to `margin` of separation on one axis
  if overlap.x < -margin || overlap.y < -margin || overlap.z < -margin {
    return
  }
  axis := 0
  if overlap.y < overlap[axis] do axis = 1
  if overlap.z < overlap[axis] do axis = 2
  pen := overlap[axis]
  // Clamp the contact rectangle to a valid region when separated
  for k in 0 ..< 3 {
    if overlap_min[k] > overlap_max[k] {
      mid := (overlap_min[k] + overlap_max[k]) * 0.5
      overlap_min[k] = mid
      overlap_max[k] = mid
    }
  }
  sign: f32 = pos_b[axis] > pos_a[axis] ? 1 : -1
  normal: [3]f32
  normal[axis] = sign
  // Contact plane at the touching faces; corners of the overlap rectangle
  plane := sign > 0 ? max_a[axis] : min_a[axis]
  u := (axis + 1) % 3
  v := (axis + 2) % 3
  m.normal = normal
  corners := [4][2]f32{
    {overlap_min[u], overlap_min[v]},
    {overlap_max[u], overlap_min[v]},
    {overlap_max[u], overlap_max[v]},
    {overlap_min[u], overlap_max[v]},
  }
  count := 0
  for c, i in corners {
    p: [3]f32
    p[axis] = plane
    p[u] = c[0]
    p[v] = c[1]
    // Degenerate overlap (edge/corner touch) produces duplicate corners; skip
    if count > 0 && linalg.length2(p - m.points[count - 1].point) < 1e-12 do continue
    m.points[count] = {
      point       = p,
      penetration = pen,
      feature_id  = u32(axis) << 4 | u32(i),
    }
    count += 1
  }
  m.count = count
  hit = count > 0
  return
}

collide_boxes :: proc(
  pos_a: [3]f32,
  rot_a: quaternion128,
  box_a: BoxCollider,
  pos_b: [3]f32,
  rot_b: quaternion128,
  box_b: BoxCollider,
  margin: f32,
) -> (m: Manifold, hit: bool) {
  if is_identity_quaternion(rot_a) && is_identity_quaternion(rot_b) {
    return collide_boxes_aligned(pos_a, box_a.half_extents, pos_b, box_b.half_extents, margin)
  }
  fa := box_frame(pos_a, rot_a, box_a.half_extents)
  fb := box_frame(pos_b, rot_b, box_b.half_extents)
  d := fb.center - fa.center

  best_pen := f32(math.F32_MAX)
  best_axis: [3]f32
  best_face_owner := -1 // 0 = A's face, 1 = B's face
  best_face_idx := -1

  // Face axes of both boxes
  for owner in 0 ..< 2 {
    f := owner == 0 ? &fa : &fb
    for i in 0 ..< 3 {
      axis := f.axes[i]
      dist := linalg.dot(d, axis)
      if dist < 0 do axis = -axis
      pen := box_extent(&fa, axis) + box_extent(&fb, axis) - math.abs(dist)
      if pen < -margin do return
      if pen < best_pen {
        best_pen = pen
        best_axis = axis
        best_face_owner = owner
        best_face_idx = i
      }
    }
  }

  // Edge-edge axes. Small relative bias prefers faces to avoid feature
  // flip-flop (box2d convention).
  EDGE_BIAS :: f32(0.95)
  EDGE_ABS_BIAS :: f32(0.01)
  best_edge_pen := f32(math.F32_MAX)
  best_edge_axis: [3]f32
  best_edge_a, best_edge_b := -1, -1
  for i in 0 ..< 3 {
    for j in 0 ..< 3 {
      axis := linalg.cross(fa.axes[i], fb.axes[j])
      len_sq := linalg.length2(axis)
      if len_sq < 1e-6 do continue // parallel edges — face axes cover this
      axis /= math.sqrt(len_sq)
      if linalg.dot(d, axis) < 0 do axis = -axis
      pen := box_extent(&fa, axis) + box_extent(&fb, axis) - math.abs(linalg.dot(d, axis))
      if pen < -margin do return
      if pen < best_edge_pen {
        best_edge_pen = pen
        best_edge_axis = axis
        best_edge_a = i
        best_edge_b = j
      }
    }
  }

  if best_edge_a >= 0 && best_edge_pen * EDGE_BIAS < best_pen - EDGE_ABS_BIAS {
    // Edge contact: closest points between the two supporting edges
    m.normal = best_edge_axis
    // Support point on A along +axis, on B along -axis
    pa := fa.center
    for i in 0 ..< 3 {
      s: f32 = linalg.dot(best_edge_axis, fa.axes[i]) > 0 ? 1 : -1
      if i == best_edge_a do continue
      pa += fa.axes[i] * (s * fa.half[i])
    }
    pb := fb.center
    for i in 0 ..< 3 {
      s: f32 = linalg.dot(best_edge_axis, fb.axes[i]) > 0 ? -1 : 1
      if i == best_edge_b do continue
      pb += fb.axes[i] * (s * fb.half[i])
    }
    // Closest points on the two edge lines
    ea := fa.axes[best_edge_a]
    eb := fb.axes[best_edge_b]
    r := pb - pa
    a_dot_b := linalg.dot(ea, eb)
    denom := 1 - a_dot_b * a_dot_b
    s_par: f32 = 0
    if math.abs(denom) > 1e-6 {
      s_par = clamp(
        (linalg.dot(r, ea) - linalg.dot(r, eb) * a_dot_b) / denom,
        -fa.half[best_edge_a],
        fa.half[best_edge_a],
      )
    }
    p_edge_a := pa + ea * s_par
    t_par := clamp(linalg.dot(p_edge_a - pb, eb), -fb.half[best_edge_b], fb.half[best_edge_b])
    p_edge_b := pb + eb * t_par
    m.count = 1
    m.points[0] = {
      point       = (p_edge_a + p_edge_b) * 0.5,
      penetration = best_edge_pen,
      feature_id  = FEATURE_EDGE_CONTACT | u32(best_edge_a) << 8 | u32(best_edge_b),
    }
    hit = true
    return
  }

  // Face contact: clip incident face against reference face side planes
  ref := best_face_owner == 0 ? &fa : &fb
  inc := best_face_owner == 0 ? &fb : &fa
  // Reference normal always points from A to B
  ref_normal := best_axis
  ref_face_normal := best_face_owner == 0 ? ref_normal : -ref_normal
  ref_sign: f32 = linalg.dot(ref.axes[best_face_idx], ref_face_normal) > 0 ? 1 : -1

  // Incident face: most anti-parallel to reference normal
  inc_idx := 0
  inc_dot := f32(math.F32_MAX)
  inc_sign: f32 = 1
  for i in 0 ..< 3 {
    dp := linalg.dot(inc.axes[i], ref_face_normal)
    if dp < inc_dot {
      inc_dot = dp
      inc_idx = i
      inc_sign = 1
    }
    if -dp < inc_dot {
      inc_dot = -dp
      inc_idx = i
      inc_sign = -1
    }
  }

  inc_verts := box_face_vertices(inc, inc_idx, inc_sign)
  poly: [8]ClipVertex
  poly_count := 4
  for v, i in inc_verts {
    poly[i] = {p = v, id = u32(inc_idx) << 16 | u32(i)}
  }

  // Clip against the 4 side planes of the reference face
  tmp: [8]ClipVertex
  for k in 0 ..< 3 {
    if k == best_face_idx do continue
    for dir in 0 ..< 2 {
      n := dir == 0 ? ref.axes[k] : -ref.axes[k]
      dd := linalg.dot(n, ref.center) + ref.half[k]
      plane_id := u32(k) << 1 | u32(dir)
      cnt := clip_polygon_plane(poly[:poly_count], n, dd, plane_id, &tmp)
      poly = tmp
      poly_count = cnt
      if poly_count == 0 do return
    }
  }

  // Keep points penetrating the reference face
  face_center := ref.center + ref.axes[best_face_idx] * (ref_sign * ref.half[best_face_idx])
  candidates: [8]ManifoldPoint
  cand_count := 0
  for i in 0 ..< poly_count {
    sep := linalg.dot(ref_face_normal, poly[i].p - face_center)
    if sep <= margin {
      candidates[cand_count] = {
        point       = poly[i].p - ref_face_normal * (sep * 0.5),
        penetration = -sep,
        feature_id  = poly[i].id | u32(best_face_owner) << 30,
      }
      cand_count += 1
    }
  }
  if cand_count == 0 do return
  m.normal = ref_normal
  reduce_manifold(&m, candidates[:cand_count])
  hit = true
  return
}

// ---------------------------------------------------------------------------
// Cylinder manifolds
// ---------------------------------------------------------------------------

// Cap-contact rim points: 4 samples on the contact circle, clipped to
// penetrating ones. Falls back to the single-point result when the contact
// is not a cap resting case.
@(private = "file")
cylinder_cap_manifold :: proc(
  m: ^Manifold,
  cyl_pos: [3]f32,
  cyl_rot: quaternion128,
  radius: f32,
  height: f32,
  plane_point: [3]f32, // point on the opposing face plane
  plane_normal: [3]f32, // toward the cylinder
  margin: f32,
) -> bool {
  axis := geometry.qy(cyl_rot)
  cap_dot := linalg.dot(axis, plane_normal)
  // Cap must be near-parallel to the face
  if math.abs(cap_dot) < 0.966 do return false // cos(15°)
  cap_sign: f32 = cap_dot > 0 ? -1 : 1
  cap_center := cyl_pos + axis * (cap_sign * height * 0.5)
  t1 := linalg.normalize(linalg.orthogonal(axis))
  t2 := linalg.cross(axis, t1)
  count := 0
  for i in 0 ..< 4 {
    dir := i == 0 ? t1 : i == 1 ? -t1 : i == 2 ? t2 : -t2
    p := cap_center + dir * radius
    sep := linalg.dot(plane_normal, p - plane_point)
    if sep >= margin do continue
    m.points[count] = {
      point       = p - plane_normal * (sep * 0.5),
      penetration = -sep,
      feature_id  = u32(i) | 0x00C10000,
    }
    count += 1
  }
  if count < 2 do return false
  m.count = count
  return true
}

collide_box_cylinder :: proc(
  pos_box: [3]f32,
  rot_box: quaternion128,
  box: BoxCollider,
  pos_cyl: [3]f32,
  rot_cyl: quaternion128,
  cyl: CylinderCollider,
  invert_normal: bool,
  margin: f32,
) -> (m: Manifold, hit: bool) {
  point, normal, penetration, single_hit := test_box_cylinder(
    pos_box, rot_box, box, pos_cyl, rot_cyl, cyl, invert_normal,
  )
  if !single_hit do return
  m = manifold_from_single_point(point, normal, penetration)
  hit = true
  // Cap resting on a box face: build a rim manifold. Normal from A to B;
  // plane normal must point toward the cylinder.
  toward_cyl := invert_normal ? -normal : normal
  // Only when the box face is the reference (normal near box face axis):
  // find penetrating rim samples against the face plane through `point`.
  if cylinder_cap_manifold(&m, pos_cyl, rot_cyl, cyl.radius, cyl.height, point, toward_cyl, margin) {
    m.normal = normal
  }
  return
}

collide_cylinders :: proc(
  pos_a: [3]f32,
  rot_a: quaternion128,
  cyl_a: CylinderCollider,
  pos_b: [3]f32,
  rot_b: quaternion128,
  cyl_b: CylinderCollider,
  margin: f32,
) -> (m: Manifold, hit: bool) {
  axis_a := geometry.qy(rot_a)
  axis_b := geometry.qy(rot_b)
  parallel := math.abs(linalg.dot(axis_a, axis_b)) > 0.99
  if parallel {
    // Choose the smaller of radial vs axial penetration — stacked cylinders
    // must resolve axially (cap contact), not sideways.
    to_b := pos_b - pos_a
    axial_dist := linalg.dot(to_b, axis_a)
    radial_vec := to_b - axis_a * axial_dist
    radial_dist := linalg.length(radial_vec)
    radius_sum := cyl_a.radius + cyl_b.radius
    half_sum := (cyl_a.height + cyl_b.height) * 0.5
    radial_pen := radius_sum - radial_dist
    axial_pen := half_sum - math.abs(axial_dist)
    if radial_pen < -margin || axial_pen < -margin do return
    if axial_pen < radial_pen {
      // Cap contact: rim points on the smaller circle around the overlap center
      normal := axial_dist > 0 ? axis_a : -axis_a
      r := min(cyl_a.radius, cyl_b.radius)
      // Circle center: midpoint of the two facing caps, shifted by radial offset
      cap_a := pos_a + normal * (cyl_a.height * 0.5)
      center := cap_a - normal * (axial_pen * 0.5)
      if radial_dist > math.F32_EPSILON {
        // Overlap circle centers differ; bias toward the overlap region
        center += radial_vec * 0.5
      }
      t1 := linalg.normalize(linalg.orthogonal(normal))
      t2 := linalg.cross(normal, t1)
      m.normal = normal
      m.count = 4
      for i in 0 ..< 4 {
        dir := i == 0 ? t1 : i == 1 ? -t1 : i == 2 ? t2 : -t2
        m.points[i] = {
          point       = center + dir * r,
          penetration = axial_pen,
          feature_id  = u32(i) | 0x00C20000,
        }
      }
      hit = true
      return
    }
    // Side contact: 2 points spanning the height overlap
    radial_dir := radial_dist > math.F32_EPSILON ? radial_vec / radial_dist : [3]f32{1, 0, 0}
    min_a := -cyl_a.height * 0.5
    max_a := cyl_a.height * 0.5
    min_b := axial_dist - cyl_b.height * 0.5
    max_b := axial_dist + cyl_b.height * 0.5
    lo := max(min_a, min_b)
    hi := min(max_a, max_b)
    contact_radial := pos_a + radial_dir * (cyl_a.radius - radial_pen * 0.5)
    m.normal = radial_dir
    m.count = 2
    m.points[0] = {
      point       = contact_radial + axis_a * lo,
      penetration = radial_pen,
      feature_id  = 0x00C30000,
    }
    m.points[1] = {
      point       = contact_radial + axis_a * hi,
      penetration = radial_pen,
      feature_id  = 0x00C30001,
    }
    hit = true
    return
  }
  point, normal, penetration, single_hit := test_cylinder_cylinder(
    pos_a, rot_a, cyl_a, pos_b, rot_b, cyl_b,
  )
  if !single_hit do return
  return manifold_from_single_point(point, normal, penetration), true
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Wrap a single-point test result whose sphere was inflated by `margin`:
// deflate the penetration back so separated-but-within-margin contacts come
// out with negative penetration (speculative).
@(private = "file")
sphere_manifold :: #force_inline proc(
  point: [3]f32,
  normal: [3]f32,
  penetration: f32,
  h: bool,
  margin: f32,
) -> (m: Manifold, hit: bool) {
  if !h do return
  return manifold_from_single_point(point, normal, penetration - margin), true
}

// margin: speculative distance — contacts are generated while shapes are
// still up to `margin` apart (penetration goes negative). The solver treats
// separated points as speculative (velocity clamp only, no push).
collide :: proc(
  collider_a: ^Collider,
  pos_a: [3]f32,
  rot_a: quaternion128,
  collider_b: ^Collider,
  pos_b: [3]f32,
  rot_b: quaternion128,
  margin: f32 = 0,
) -> (m: Manifold, hit: bool) {
  switch shape_a in collider_a {
  case FanCollider:
    return
  case SphereCollider:
    inflated := SphereCollider{radius = shape_a.radius + margin}
    switch shape_b in collider_b {
    case FanCollider:
      return
    case SphereCollider:
      return sphere_manifold(test_sphere_sphere(pos_a, inflated, pos_b, shape_b), margin)
    case BoxCollider:
      return sphere_manifold(test_box_sphere(pos_b, rot_b, shape_b, pos_a, inflated, invert_normal = true), margin)
    case CylinderCollider:
      return sphere_manifold(test_sphere_cylinder(pos_a, inflated, pos_b, rot_b, shape_b), margin)
    }
  case BoxCollider:
    switch shape_b in collider_b {
    case FanCollider:
      return
    case SphereCollider:
      inflated := SphereCollider{radius = shape_b.radius + margin}
      return sphere_manifold(test_box_sphere(pos_a, rot_a, shape_a, pos_b, inflated), margin)
    case BoxCollider:
      return collide_boxes(pos_a, rot_a, shape_a, pos_b, rot_b, shape_b, margin)
    case CylinderCollider:
      return collide_box_cylinder(pos_a, rot_a, shape_a, pos_b, rot_b, shape_b, invert_normal = false, margin = margin)
    }
  case CylinderCollider:
    switch shape_b in collider_b {
    case FanCollider:
      return
    case SphereCollider:
      inflated := SphereCollider{radius = shape_b.radius + margin}
      return sphere_manifold(test_sphere_cylinder(pos_b, inflated, pos_a, rot_a, shape_a, invert_normal = true), margin)
    case BoxCollider:
      return collide_box_cylinder(pos_b, rot_b, shape_b, pos_a, rot_a, shape_a, invert_normal = true, margin = margin)
    case CylinderCollider:
      return collide_cylinders(pos_a, rot_a, shape_a, pos_b, rot_b, shape_b, margin)
    }
  }
  return
}

collide_bodies :: #force_inline proc(body_a: ^$A, body_b: ^$B, margin: f32 = 0) -> (Manifold, bool) {
  return collide(
    &body_a.collider,
    body_a.position,
    body_a.rotation,
    &body_b.collider,
    body_b.position,
    body_b.rotation,
    margin,
  )
}
