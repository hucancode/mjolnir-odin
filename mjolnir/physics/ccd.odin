package physics

import "../geometry"
import "core:math"
import "core:math/linalg"

// Time of Impact result
TOIResult :: struct {
  has_impact: bool,
  time:       f32, // 0.0 to 1.0 (fraction of motion)
  normal:     [3]f32,
  point:      [3]f32,
}

swept_sphere_sphere :: proc(
  pos_a, pos_b: [3]f32,
  radius_a, radius_b: f32,
  velocity_a: [3]f32,
) -> TOIResult {
  result: TOIResult
  // Relative motion
  motion := velocity_a
  motion_length_sq := linalg.length2(motion)
  if motion_length_sq < math.F32_EPSILON {
    // Not moving - use discrete test
    delta := pos_b - pos_a
    distance_sq := linalg.length2(delta)
    radius_sum := radius_a + radius_b
    if distance_sq < radius_sum * radius_sum {
      result.has_impact = true
      result.time = 0.0
      distance := math.sqrt(distance_sq)
      result.normal =
        distance > math.F32_EPSILON ? delta / distance : linalg.VECTOR3F32_Y_AXIS
      result.point = pos_a + result.normal * radius_a
    }
    return result
  }
  // Solve quadratic: |pos_a + t*velocity - pos_b|^2 = (radius_a + radius_b)^2
  // Let d = pos_a - pos_b
  // (d + t*v)·(d + t*v) = r^2
  // v·v*t^2 + 2*d·v*t + d·d - r^2 = 0
  d := pos_a - pos_b
  radius_sum := radius_a + radius_b
  a := motion_length_sq
  b := 2.0 * linalg.dot(d, motion)
  c := linalg.length2(d) - radius_sum * radius_sum
  discriminant := b * b - 4.0 * a * c
  if discriminant < 0 {
    // No intersection
    return result
  }
  // Take earlier root (first impact)
  sqrt_disc := math.sqrt(discriminant)
  t1 := (-b - sqrt_disc) / (2.0 * a)
  t2 := (-b + sqrt_disc) / (2.0 * a)
  // We want the first impact in the range [0, 1]
  t := t1
  if t < 0 {
    t = t2
  }
  if t >= 0 && t <= 1.0 {
    result.has_impact = true
    result.time = t
    impact_pos_a := pos_a + motion * t
    delta := pos_b - impact_pos_a
    distance := linalg.length(delta)
    result.normal =
      distance > math.F32_EPSILON ? delta / distance : linalg.VECTOR3F32_Y_AXIS
    result.point = impact_pos_a + result.normal * radius_a
  }
  return result
}

swept_sphere_box :: proc(
  center: [3]f32,
  radius: f32,
  velocity: [3]f32,
  box_min, box_max: [3]f32,
) -> TOIResult {
  result: TOIResult
  // Expand the box by sphere radius - now ray vs expanded box
  expanded_min := box_min - [3]f32{radius, radius, radius}
  expanded_max := box_max + [3]f32{radius, radius, radius}
  // Ray-AABB intersection (Slab method)
  t_min := f32(-1e6)
  t_max := f32(1e6)
  hit_normal := linalg.VECTOR3F32_Y_AXIS
  #unroll for i in 0 ..< 3 {
    if abs(velocity[i]) < math.F32_EPSILON {
      // Ray parallel to slab
      if center[i] < expanded_min[i] || center[i] > expanded_max[i] {
        return result // No hit
      }
    } else {
      // Compute intersection with slab
      inv_d := 1.0 / velocity[i]
      t1 := (expanded_min[i] - center[i]) * inv_d
      t2 := (expanded_max[i] - center[i]) * inv_d
      // Determine which is near/far
      if t1 > t2 {
        t1, t2 = t2, t1
      }
      // Update intervals
      if t1 > t_min {
        t_min = t1
        // Normal points from box to sphere
        hit_normal = {}
        hit_normal[i] = -linalg.sign(velocity[i])
      }
      if t2 < t_max {
        t_max = t2
      }
      // Early exit if no overlap
      if t_min > t_max {
        return result
      }
    }
  }
  // Check if impact is in valid range [0, 1]
  if t_min >= 0 && t_min <= 1.0 {
    result.has_impact = true
    result.time = t_min
    result.normal = hit_normal
    result.point = center + velocity * t_min
  }
  return result
}

swept_box_box :: proc(
  pos_a, pos_b: [3]f32,
  half_extents_a, half_extents_b: [3]f32,
  velocity_a: [3]f32,
) -> TOIResult {
  result: TOIResult
  // Minkowski sum: treat as a point (pos_a) moving toward an expanded box
  // The expanded box has size = half_extents_a + half_extents_b
  sum_half_extents := half_extents_a + half_extents_b
  expanded_min := pos_b - sum_half_extents
  expanded_max := pos_b + sum_half_extents
  // Ray-AABB intersection (Slab method)
  t_min := f32(-1e6)
  t_max := f32(1e6)
  hit_normal := linalg.VECTOR3F32_Y_AXIS
  #unroll for i in 0 ..< 3 {
    if abs(velocity_a[i]) < math.F32_EPSILON {
      // Ray parallel to slab
      if pos_a[i] < expanded_min[i] || pos_a[i] > expanded_max[i] {
        return result // No hit
      }
    } else {
      // Compute intersection with slab
      inv_d := 1.0 / velocity_a[i]
      t1 := (expanded_min[i] - pos_a[i]) * inv_d
      t2 := (expanded_max[i] - pos_a[i]) * inv_d
      // Determine which is near/far
      if t1 > t2 {
        t1, t2 = t2, t1
      }
      // Update intervals
      if t1 > t_min {
        t_min = t1
        // Normal points from B to A
        hit_normal = {}
        hit_normal[i] = -linalg.sign(velocity_a[i])
      }
      if t2 < t_max {
        t_max = t2
      }
      // Early exit if no overlap
      if t_min > t_max {
        return result
      }
    }
  }
  // Check if impact is in valid range [0, 1]
  if t_min >= 0 && t_min <= 1.0 {
    result.has_impact = true
    result.time = t_min
    result.normal = hit_normal
    result.point = pos_a + velocity_a * t_min
  }
  return result
}

// Conservative sweep radius for the sphere-sphere fallback: exact for
// spheres, radial for cylinders (heights ignored, same as narrowphase
// approximation), bounding sphere for boxes.
@(private = "file")
swept_radius :: proc(c: ^Collider) -> f32 {
  switch s in c {
  case SphereCollider:
    return s.radius
  case BoxCollider:
    return linalg.length(s.half_extents)
  case CylinderCollider:
    return s.radius
  case FanCollider:
    return 0
  }
  return 0
}

// TOI of A moving by velocity_a against a stationary B. Exact slab tests for
// sphere/cylinder-vs-aligned-box and aligned box-box; everything else falls
// back to a conservative swept-sphere approximation.
swept_test :: proc(
  collider_a, collider_b: ^Collider,
  pos_a, pos_b: [3]f32,
  rot_a, rot_b: quaternion128,
  velocity_a: [3]f32,
) -> TOIResult {
  if _, a_is_fan := collider_a^.(FanCollider); a_is_fan do return {} // fans are trigger-only
  if _, b_is_fan := collider_b^.(FanCollider); b_is_fan do return {}
  #partial switch shape_a in collider_a {
  case SphereCollider:
    if box_b, is_box := collider_b^.(BoxCollider); is_box {
      return swept_sphere_box(
        pos_a, shape_a.radius, velocity_a,
        pos_b - box_b.half_extents, pos_b + box_b.half_extents,
      )
    }
  case CylinderCollider:
    if box_b, is_box := collider_b^.(BoxCollider); is_box {
      return swept_sphere_box(
        pos_a, shape_a.radius, velocity_a,
        pos_b - box_b.half_extents, pos_b + box_b.half_extents,
      )
    }
  case BoxCollider:
    if sph_b, is_sphere := collider_b^.(SphereCollider); is_sphere {
      result := swept_sphere_box(
        pos_b, sph_b.radius, -velocity_a,
        pos_a - shape_a.half_extents, pos_a + shape_a.half_extents,
      )
      if result.has_impact do result.normal = -result.normal
      return result
    }
    box_b, is_box := collider_b^.(BoxCollider)
    if is_box && is_identity_quaternion(rot_a) && is_identity_quaternion(rot_b) {
      return swept_box_box(pos_a, pos_b, shape_a.half_extents, box_b.half_extents, velocity_a)
    }
  }
  return swept_sphere_sphere(
    pos_a, pos_b,
    swept_radius(collider_a), swept_radius(collider_b),
    velocity_a,
  )
}
