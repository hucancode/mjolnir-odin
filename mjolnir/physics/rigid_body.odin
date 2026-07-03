package physics

import "../geometry"
import "core:math"
import "core:math/linalg"

RigidBody :: struct {
  position:             [3]f32,
  cached_sphere_radius: f32,
  rotation:             quaternion128,
  collider:             Collider,
  cached_aabb:          geometry.Aabb,
  cached_sphere_center: [3]f32,
}

TriggerBody :: struct {
  using base: RigidBody,
}

StaticRigidBody :: struct {
  using base:   RigidBody,
  restitution:  f32,
  friction:     f32,
}

DynamicRigidBody :: struct {
  using body:              StaticRigidBody,
  enable_rotation:         bool,
  is_sleeping:             bool,
  inv_inertia:             [3]f32, // Diagonal inverse inertia in body space
  inv_inertia_world:       matrix[3, 3]f32,
  velocity:                [3]f32,
  inv_mass:                f32,
  angular_velocity:        [3]f32,
  linear_damping:          f32,
  force:                   [3]f32,
  angular_damping:         f32,
  torque:                  [3]f32,
  sleep_timer:             f32,
  is_killed:               bool,
  // Position at frame start (after CCD, before substeps) — the solver
  // recovers per-substep contact separation from position - position0.
  position0:               [3]f32,
}

static_rigid_body_init :: proc(
  self: ^StaticRigidBody,
  position: [3]f32 = {0, 0, 0},
  rotation := linalg.QUATERNIONF32_IDENTITY,
) {
  self.position = position
  self.rotation = rotation
  self.restitution = 0.2
  self.friction = 0.8
}

rigid_body_init :: proc(
  self: ^DynamicRigidBody,
  position: [3]f32 = {0, 0, 0},
  rotation := linalg.QUATERNIONF32_IDENTITY,
  mass: f32 = 1.0,
  enable_rotation := true,
) {
  self.position = position
  self.rotation = rotation
  self.inv_mass = 1.0 / mass
  self.restitution = 0.2
  self.friction = 0.8
  self.linear_damping = 0.01
  self.angular_damping = 0.05
  self.enable_rotation = enable_rotation
  self.inv_inertia = {1.0, 1.0, 1.0}
  self.is_sleeping = false
  self.sleep_timer = 0.0
  update_world_inertia(self)
}

skew :: #force_inline proc "contextless" (v: [3]f32) -> matrix[3, 3]f32 {
  return matrix[3, 3]f32{
    0, -v.z, v.y,
    v.z, 0, -v.x,
    -v.y, v.x, 0,
  }
}

// Implicit gyroscopic torque: one Newton-Raphson iteration on
// I*(w2−w1) + h*(w2 * I*w2) = 0 in body frame. Slightly
// dissipative but unconditionally stable — energy can only decrease.
solve_gyroscopic :: proc(body: ^DynamicRigidBody, dt: f32) {
  ii := body.inv_inertia
  if ii.x <= 0 || ii.y <= 0 || ii.z <= 0 do return
  q := body.rotation
  w1 := linalg.quaternion128_mul_vector3(
    linalg.quaternion_inverse(q),
    body.angular_velocity,
  )
  inertia := 1.0 / ii
  iw := inertia * w1
  f := dt * linalg.cross(w1, iw)
  i_mat := matrix[3, 3]f32{
    inertia.x, 0, 0,
    0, inertia.y, 0,
    0, 0, inertia.z,
  }
  jac := i_mat + dt * (skew(w1) * i_mat - skew(iw))
  w2 := w1 - linalg.inverse(jac) * f
  body.angular_velocity = linalg.quaternion128_mul_vector3(q, w2)
}

update_world_inertia :: proc(self: ^DynamicRigidBody) {
  if !self.enable_rotation {
    self.inv_inertia_world = {}
    return
  }
  r := linalg.matrix3_from_quaternion(self.rotation)
  d := matrix[3, 3]f32{
    self.inv_inertia.x, 0, 0,
    0, self.inv_inertia.y, 0,
    0, 0, self.inv_inertia.z,
  }
  self.inv_inertia_world = r * d * linalg.transpose(r)
}

wake_up :: #force_inline proc(self: ^DynamicRigidBody) {
  self.is_sleeping = false
  self.sleep_timer = 0.0
}

set_mass :: proc(self: ^DynamicRigidBody, mass: f32) {
  if mass <= 0.0 do return
  new_inv_mass := 1.0 / mass
  if self.inv_mass > 0.0 {
    mass_ratio := new_inv_mass / self.inv_mass
    self.inv_inertia = mass_ratio * self.inv_inertia
  }
  self.inv_mass = new_inv_mass
  update_world_inertia(self)
}

set_box_inertia :: proc(self: ^DynamicRigidBody, half_extents: [3]f32) {
  v := half_extents * half_extents
  inv_inertia_factor := 3.0 * self.inv_mass
  self.inv_inertia = inv_inertia_factor / [3]f32{v.y + v.z, v.x + v.z, v.x + v.y}
  update_world_inertia(self)
}

set_sphere_inertia :: proc(self: ^DynamicRigidBody, radius: f32) {
  inv_i := (5.0 / 2.0) * self.inv_mass / (radius * radius)
  self.inv_inertia = {inv_i, inv_i, inv_i}
  update_world_inertia(self)
}

set_cylinder_inertia :: proc(
  self: ^DynamicRigidBody,
  radius: f32,
  height: f32,
) {
  r2 := radius * radius
  h2 := height * height
  inv_ix := 12.0 * self.inv_mass / (3.0 * r2 + h2)
  inv_iy := 2.0 * self.inv_mass / r2
  self.inv_inertia = [3]f32{inv_ix, inv_iy, inv_ix}
  update_world_inertia(self)
}

set_inertia_from_collider :: proc(self: ^DynamicRigidBody, collider: Collider) {
  switch c in collider {
  case SphereCollider:   set_sphere_inertia(self, c.radius)
  case BoxCollider:      set_box_inertia(self, c.half_extents)
  case CylinderCollider: set_cylinder_inertia(self, c.radius, c.height)
  case FanCollider:      set_cylinder_inertia(self, c.radius, c.height)
  }
}

apply_force :: proc(self: ^DynamicRigidBody, force: [3]f32) {
  self.force += force
  wake_up(self)
}

apply_force_at_point :: proc(
  self: ^DynamicRigidBody,
  force: [3]f32,
  point: [3]f32,
  center: [3]f32,
) {
  self.force += force
  if !self.enable_rotation do return
  r := point - center
  self.torque += linalg.cross(r, force)
  wake_up(self)
}

apply_impulse :: proc(self: ^DynamicRigidBody, impulse: [3]f32) {
  self.velocity += impulse * self.inv_mass
  wake_up(self)
}

apply_impulse_at_point :: #force_inline proc(
  self: ^DynamicRigidBody,
  impulse: [3]f32,
  point: [3]f32,
) {
  apply_impulse_at_point_no_wake(self, impulse, point)
  wake_up(self)
}

// Solver-internal impulse application. Caller must have already woken the body.
// Skips the wake_up writes that pollute the cache during constraint iteration.
apply_impulse_at_point_no_wake :: #force_inline proc(
  self: ^DynamicRigidBody,
  impulse: [3]f32,
  point: [3]f32,
) {
  self.velocity += impulse * self.inv_mass
  if !self.enable_rotation do return
  r := point - self.position
  angular_impulse := linalg.cross(r, impulse)
  self.angular_velocity += self.inv_inertia_world * angular_impulse
}

update_cached_aabb :: proc(self: ^RigidBody) {
  self.cached_aabb = collider_calculate_aabb(
    &self.collider,
    self.position,
    self.rotation,
  )
  self.cached_sphere_center = geometry.aabb_center(self.cached_aabb)
  switch sh in self.collider {
  case SphereCollider:
    self.cached_sphere_radius = sh.radius
  case BoxCollider:
    self.cached_sphere_radius = linalg.length(sh.half_extents)
  case CylinderCollider:
    h := sh.height * 0.5
    self.cached_sphere_radius = math.sqrt(sh.radius * sh.radius + h * h)
  case FanCollider:
    h := sh.height * 0.5
    self.cached_sphere_radius = math.sqrt(sh.radius * sh.radius + h * h)
  }
}
