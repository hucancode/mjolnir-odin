package physics

import "../geometry"
import "core:math"
import "core:math/linalg"

is_identity_quaternion :: proc "contextless" (q: quaternion128) -> bool {
  epsilon :: 1e-6
  return(
    math.abs(q.x) < epsilon &&
    math.abs(q.y) < epsilon &&
    math.abs(q.z) < epsilon &&
    math.abs(q.w - 1.0) < epsilon \
  )
}

// One solver point of a contact manifold
// base_separation: signed separation at prepare time (negative = penetrating);
// current separation is recovered per substep from body position deltas.
// relative_velocity: normal approach speed captured at prepare, used by the
// post-substep restitution pass.
ContactPoint :: struct {
  point:              [3]f32,
  penetration:        f32, // filled by narrowphase; prepare converts to base_separation
  feature_id:         u32,
  normal_impulse:     f32,
  max_normal_impulse: f32,
  normal_mass:        f32,
  base_separation:    f32,
  relative_velocity:  f32,
  r_a:                [3]f32,
  r_b:                [3]f32,
}

// Contact between two dynamic bodies: up to 4 normal points plus one
// manifold-level friction constraint (coupled 2D tangent at the centroid +
// twist about the normal).
DynamicContact :: struct {
  body_a:          DynamicRigidBodyHandle,
  body_b:          DynamicRigidBodyHandle,
  normal:          [3]f32,
  restitution:     f32,
  friction:        f32,
  count:           int,
  points:          [MAX_MANIFOLD_POINTS]ContactPoint,
  tangent1:        [3]f32,
  tangent2:        [3]f32,
  r_a_c:           [3]f32, // centroid offset from body A center
  r_b_c:           [3]f32,
  tangent_impulse: [2]f32,
  twist_impulse:   f32,
  tangent_mass:    matrix[2, 2]f32, // inverse of the coupled tangent K matrix
  twist_mass:      f32,
}

// Contact between dynamic body (A) and static body (B)
StaticContact :: struct {
  body_a:          DynamicRigidBodyHandle,
  body_b:          StaticRigidBodyHandle,
  normal:          [3]f32,
  restitution:     f32,
  friction:        f32,
  count:           int,
  points:          [MAX_MANIFOLD_POINTS]ContactPoint,
  tangent1:        [3]f32,
  tangent2:        [3]f32,
  r_a_c:           [3]f32,
  tangent_impulse: [2]f32,
  twist_impulse:   f32,
  tangent_mass:    matrix[2, 2]f32,
  twist_mass:      f32,
}

// Warmstart pair keys include handle generations so a recycled slot never
// inherits the previous occupant's impulses.
collision_pair_hash_dynamic :: proc "contextless" (
  body_a: DynamicRigidBodyHandle,
  body_b: DynamicRigidBodyHandle,
) -> u64 {
  ha := u64(body_a.index) | (u64(body_a.generation) << 32)
  hb := u64(body_b.index) | (u64(body_b.generation) << 32)
  return ha * 0x9E3779B97F4A7C15 ~ (hb + 0x517CC1B727220A95)
}

collision_pair_hash_static :: proc "contextless" (
  body_a: DynamicRigidBodyHandle,
  body_b: StaticRigidBodyHandle,
) -> u64 {
  ha := u64(body_a.index) | (u64(body_a.generation) << 32)
  hb := u64(body_b.index) | (u64(body_b.generation) << 32)
  return ha * 0x9E3779B97F4A7C15 ~ (hb + 0xD1B54A32D192ED03)
}

mix_friction :: #force_inline proc "contextless" (a, b: f32) -> f32 {
  return math.sqrt(a * b)
}

mix_restitution :: #force_inline proc "contextless" (a, b: f32) -> f32 {
  return max(a, b)
}

collision_pair_hash :: proc {
  collision_pair_hash_dynamic,
  collision_pair_hash_static,
}

ContactWarmstart :: struct {
  count:           int,
  feature_ids:     [MAX_MANIFOLD_POINTS]u32,
  normal_impulses: [MAX_MANIFOLD_POINTS]f32,
  tangent:         [2]f32,
  twist:           f32,
}

contact_has_impulse :: #force_inline proc(c: ^$T) -> bool {
  if c.tangent_impulse != {0, 0} || c.twist_impulse != 0 do return true
  for i in 0 ..< c.count {
    if c.points[i].normal_impulse != 0 do return true
  }
  return false
}

contact_store_warmstart :: proc(c: ^$T) -> (w: ContactWarmstart) {
  w.count = c.count
  for i in 0 ..< c.count {
    w.feature_ids[i] = c.points[i].feature_id
    w.normal_impulses[i] = c.points[i].normal_impulse
  }
  w.tangent = c.tangent_impulse
  w.twist = c.twist_impulse
  return
}

contact_apply_warmstart :: proc(c: ^$T, w: ^ContactWarmstart) {
  for i in 0 ..< c.count {
    for j in 0 ..< w.count {
      if c.points[i].feature_id == w.feature_ids[j] {
        c.points[i].normal_impulse = w.normal_impulses[j]
        break
      }
    }
  }
  c.tangent_impulse = w.tangent
  c.twist_impulse = w.twist
}

// Fast bounding sphere intersection test (use before expensive narrow phase)
bounding_spheres_intersect :: proc "contextless" (
  pos_a: [3]f32,
  radius_a: f32,
  pos_b: [3]f32,
  radius_b: f32,
) -> bool {
  delta := pos_b - pos_a
  dist_sq := linalg.dot(delta, delta)
  radius_sum := radius_a + radius_b
  return dist_sq <= radius_sum * radius_sum
}

test_sphere_sphere :: proc(
  pos_a: [3]f32,
  sphere_a: SphereCollider,
  pos_b: [3]f32,
  sphere_b: SphereCollider,
) -> (
  point: [3]f32,
  normal: [3]f32,
  penetration: f32,
  hit: bool,
) {
  delta := pos_b - pos_a
  distance_sq := linalg.length2(delta)
  radius_sum := sphere_a.radius + sphere_b.radius
  if distance_sq > radius_sum * radius_sum {
    return
  }
  distance := math.sqrt(distance_sq)
  normal =
    distance > math.F32_EPSILON ? delta / distance : linalg.VECTOR3F32_Y_AXIS
  penetration = radius_sum - distance
  point = pos_a + normal * (sphere_a.radius - penetration * 0.5)
  hit = true
  return
}

test_box_sphere :: proc(
  pos_box: [3]f32,
  rot_box: quaternion128,
  box: BoxCollider,
  pos_sphere: [3]f32,
  sphere: SphereCollider,
  invert_normal: bool = false,
) -> (
  closest: [3]f32,
  normal: [3]f32,
  penetration: f32,
  hit: bool,
) {
  is_aligned := is_identity_quaternion(rot_box)
  if is_aligned {
    min_box := pos_box - box.half_extents
    max_box := pos_box + box.half_extents
    closest = linalg.clamp(pos_sphere, min_box, max_box)
    delta := pos_sphere - closest
    distance_sq := linalg.length2(delta)
    if distance_sq > sphere.radius * sphere.radius {
      return
    }
    distance := math.sqrt(distance_sq)
    normal =
      distance > math.F32_EPSILON ? delta / distance : linalg.VECTOR3F32_Y_AXIS
    if invert_normal do normal = -normal
    penetration = sphere.radius - distance
    hit = true
    return
  }
  obb := geometry.Obb {
    center       = pos_box,
    half_extents = box.half_extents,
    rotation     = rot_box,
  }
  closest, normal, penetration, hit = geometry.obb_sphere_intersect(obb, pos_sphere, sphere.radius)
  if invert_normal && hit do normal = -normal
  return
}

// Point-in-cylinder test - checks if point is inside a cylinder
test_point_cylinder :: proc(
  point: [3]f32,
  cylinder_center: [3]f32,
  cylinder_rot: quaternion128,
  cylinder: CylinderCollider,
) -> bool {
  // Transform point to cylinder's local space
  local_point := point - cylinder_center
  // Rotate point by inverse of cylinder's rotation
  inv_rot := linalg.quaternion_inverse(cylinder_rot)
  local_point = geometry.qmv(inv_rot, local_point)
  // In local space, cylinder axis is Y, check radial distance and height
  half_height := cylinder.height * 0.5
  return(
    linalg.length2(local_point.xz) <= cylinder.radius * cylinder.radius &&
    math.abs(local_point.y) <= half_height \
  )
}

// Point-in-fan test - checks if point is inside a fan (partial cylinder)
test_point_fan :: proc(
  point: [3]f32,
  fan_center: [3]f32,
  fan_rot: quaternion128,
  fan: FanCollider,
) -> bool {
  // Transform point to fan's local space
  local_point := point - fan_center
  // Rotate point by inverse of fan's rotation
  inv_rot := linalg.quaternion_inverse(fan_rot)
  local_point = geometry.qmv(inv_rot, local_point)
  // In local space, fan forward direction is +Z, axis is Y
  // Check radial distance and height first (like cylinder)
  radial_dist_sq := linalg.length2(local_point.xz)
  half_height := fan.height * 0.5
  if radial_dist_sq > fan.radius * fan.radius ||
     math.abs(local_point.y) > half_height {
    return false
  }
  // Check if point is within the fan's angular range
  // Forward is +Z, so we measure angle from +Z axis in XZ plane
  if radial_dist_sq < math.F32_EPSILON {
    return true // point is on the axis
  }
  angle_from_forward := math.atan2(local_point.x, local_point.z)
  half_angle := fan.angle * 0.5
  return math.abs(angle_from_forward) <= half_angle
}

test_sphere_cylinder :: proc(
  pos_sphere: [3]f32,
  sphere: SphereCollider,
  pos_cylinder: [3]f32,
  rot_cylinder: quaternion128,
  cylinder: CylinderCollider,
  invert_normal: bool = false,
) -> (
  point: [3]f32,
  normal: [3]f32,
  penetration: f32,
  hit: bool,
) {
  // Transform sphere to cylinder's local space
  to_sphere := pos_sphere - pos_cylinder
  inv_rot := linalg.quaternion_inverse(rot_cylinder)
  local_sphere := geometry.qmv(inv_rot, to_sphere)
  // In local space, cylinder axis is Y
  half_height := cylinder.height * 0.5
  // Vector from axis to sphere center in XZ plane
  radial := [3]f32{local_sphere.x, 0, local_sphere.z}
  radial_dist := linalg.length(radial)
  // Determine which region the sphere center is in
  above_cylinder := local_sphere.y > half_height
  below_cylinder := local_sphere.y < -half_height
  inside_height := !above_cylinder && !below_cylinder
  inside_radial := radial_dist < cylinder.radius
  // Find closest point on cylinder surface
  local_closest: [3]f32
  if above_cylinder || below_cylinder {
    // Sphere center is above or below the cylinder
    cap_y := above_cylinder ? half_height : -half_height
    if inside_radial {
      // Closest point is on the cap (directly above/below sphere)
      local_closest = [3]f32{local_sphere.x, cap_y, local_sphere.z}
    } else {
      // Closest point is on the rim (edge of cap)
      radial_dir := radial / radial_dist
      local_closest = [3]f32 {
        radial_dir.x * cylinder.radius,
        cap_y,
        radial_dir.z * cylinder.radius,
      }
    }
  } else if inside_radial {
    // Sphere center is INSIDE the cylinder - find minimum penetration axis
    dist_to_top := half_height - local_sphere.y
    dist_to_bottom := local_sphere.y + half_height
    dist_to_side := cylinder.radius - radial_dist
    if dist_to_side <= dist_to_top && dist_to_side <= dist_to_bottom {
      // Push out through curved surface
      if radial_dist < math.F32_EPSILON {
        local_closest = [3]f32{cylinder.radius, local_sphere.y, 0}
      } else {
        radial_dir := radial / radial_dist
        local_closest = [3]f32 {
          radial_dir.x * cylinder.radius,
          local_sphere.y,
          radial_dir.z * cylinder.radius,
        }
      }
    } else if dist_to_top <= dist_to_bottom {
      // Push out through top cap
      local_closest = [3]f32{local_sphere.x, half_height, local_sphere.z}
    } else {
      // Push out through bottom cap
      local_closest = [3]f32{local_sphere.x, -half_height, local_sphere.z}
    }
  } else {
    // Sphere center is outside radial extent - closest on curved surface
    radial_dir := radial / radial_dist
    clamped_y := linalg.clamp(local_sphere.y, -half_height, half_height)
    local_closest = [3]f32 {
      radial_dir.x * cylinder.radius,
      clamped_y,
      radial_dir.z * cylinder.radius,
    }
  }
  // Compute surface normal based on which surface the closest point is on
  local_normal: [3]f32
  if local_closest.y >= half_height - math.F32_EPSILON {
    // On top cap
    local_normal = {0, 1, 0}
  } else if local_closest.y <= -half_height + math.F32_EPSILON {
    // On bottom cap
    local_normal = {0, -1, 0}
  } else {
    // On curved surface
    local_normal = linalg.normalize(
      [3]f32{local_closest.x, 0, local_closest.z},
    )
  }
  // Transform to world space
  world_closest := pos_cylinder + geometry.qmv(rot_cylinder, local_closest)
  world_normal := geometry.qmv(rot_cylinder, local_normal)
  delta := pos_sphere - world_closest
  dist_sq := linalg.length2(delta)
  if dist_sq > sphere.radius * sphere.radius {
    return
  }
  distance := math.sqrt(dist_sq)
  // Use surface normal (more stable than delta-based normal when deeply penetrating)
  normal = world_normal
  penetration = sphere.radius - distance
  // Contact point is between the surfaces
  point = world_closest + normal * (penetration * 0.5)
  // Invert normal for collision response direction
  // Convention: normal points FROM body_a TO body_b
  // When invert_normal=true, cylinder is body_a, so normal should point toward sphere (negate surface normal)
  if !invert_normal do normal = -normal
  hit = true
  return
}

test_box_cylinder :: proc(
  pos_box: [3]f32,
  rot_box: quaternion128,
  box: BoxCollider,
  pos_cylinder: [3]f32,
  rot_cylinder: quaternion128,
  cylinder: CylinderCollider,
  invert_normal: bool = false,
) -> (
  point: [3]f32,
  normal: [3]f32,
  penetration: f32,
  hit: bool,
) {
  obb := geometry.Obb {
    center       = pos_box,
    half_extents = box.half_extents,
    rotation     = rot_box,
  }
  point, normal, penetration, hit = geometry.obb_cylinder_intersect(
    obb,
    pos_cylinder,
    rot_cylinder,
    cylinder.radius,
    cylinder.height,
  )
  if invert_normal && hit do normal = -normal
  return
}

test_cylinder_cylinder :: proc(
  pos_a: [3]f32,
  rot_a: quaternion128,
  cylinder_a: CylinderCollider,
  pos_b: [3]f32,
  rot_b: quaternion128,
  cylinder_b: CylinderCollider,
) -> (
  point: [3]f32,
  normal: [3]f32,
  penetration: f32,
  hit: bool,
) {
  // Transform cylinder B to cylinder A's local space
  to_b := pos_b - pos_a
  inv_rot_a := linalg.quaternion_inverse(rot_a)
  local_b_center := geometry.qmv(inv_rot_a, to_b)
  // Cylinder B's axis in cylinder A's local space
  b_axis_world := geometry.qy(rot_b)
  b_axis_local := geometry.qmv(inv_rot_a, b_axis_world)
  // Check if axes are parallel
  parallel := math.abs(math.abs(b_axis_local.y) - 1.0) < 0.01
  if parallel {
    // Axes are parallel - treat as 2D circle-circle in XZ plane
    radial := [3]f32{local_b_center.x, 0, local_b_center.z}
    radial_dist := linalg.length(radial)
    radius_sum := cylinder_a.radius + cylinder_b.radius
    if radial_dist > radius_sum {
      return
    }
    // Check height overlap
    half_height_a := cylinder_a.height * 0.5
    half_height_b := cylinder_b.height * 0.5
    min_a := -half_height_a
    max_a := half_height_a
    min_b := local_b_center.y - half_height_b
    max_b := local_b_center.y + half_height_b
    if max_a < min_b || min_a > max_b {
      return
    }
    // Collision detected
    radial_dir :=
      radial_dist > math.F32_EPSILON ? radial / radial_dist : [3]f32{1, 0, 0}
    local_normal := radial_dir
    local_point :=
      local_normal * (cylinder_a.radius - (radius_sum - radial_dist) * 0.5)
    // Transform back to world space
    normal = geometry.qmv(rot_a, local_normal)
    point = pos_a + geometry.qmv(rot_a, local_point)
    penetration = radius_sum - radial_dist
    hit = true
    return
  }
  // Non-parallel cylinders - use capsule-capsule approximation
  // More accurate than sphere approximation while remaining fast
  axis_a := geometry.qy(rot_a)
  axis_b := geometry.qy(rot_b)

  half_height_a := cylinder_a.height * 0.5
  half_height_b := cylinder_b.height * 0.5

  // Compute capsule endpoints
  p0_a := pos_a - axis_a * half_height_a
  p1_a := pos_a + axis_a * half_height_a
  p0_b := pos_b - axis_b * half_height_b
  p1_b := pos_b + axis_b * half_height_b

  // Find closest points between two line segments (capsule axes)
  d1 := p1_a - p0_a
  d2 := p1_b - p0_b
  r := p0_a - p0_b

  a := linalg.dot(d1, d1)
  e := linalg.dot(d2, d2)
  f := linalg.dot(d2, r)

  s, t: f32

  // Check if either or both segments degenerate into points
  if a <= math.F32_EPSILON && e <= math.F32_EPSILON {
    s = 0.0
    t = 0.0
  } else if a <= math.F32_EPSILON {
    s = 0.0
    t = clamp(f / e, 0.0, 1.0)
  } else {
    c := linalg.dot(d1, r)
    if e <= math.F32_EPSILON {
      t = 0.0
      s = clamp(-c / a, 0.0, 1.0)
    } else {
      b := linalg.dot(d1, d2)
      denom := a * e - b * b

      if denom != 0.0 {
        s = clamp((b * f - c * e) / denom, 0.0, 1.0)
      } else {
        s = 0.0
      }

      t = (b * s + f) / e

      if t < 0.0 {
        t = 0.0
        s = clamp(-c / a, 0.0, 1.0)
      } else if t > 1.0 {
        t = 1.0
        s = clamp((b - c) / a, 0.0, 1.0)
      }
    }
  }

  // Compute closest points
  c1 := p0_a + d1 * s
  c2 := p0_b + d2 * t

  // Check distance between closest points
  delta := c2 - c1
  dist_sq := linalg.length2(delta)
  radius_sum := cylinder_a.radius + cylinder_b.radius

  if dist_sq > radius_sum * radius_sum {
    return
  }

  distance := math.sqrt(dist_sq)
  normal = distance > math.F32_EPSILON ? delta / distance : linalg.VECTOR3F32_Y_AXIS
  penetration = radius_sum - distance
  point = c1 + normal * (cylinder_a.radius - penetration * 0.5)
  hit = true
  return
}

