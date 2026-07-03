package main

import "../../mjolnir"
import "../../mjolnir/geometry"
import "../../mjolnir/physics"
import "../../mjolnir/world"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import mu "vendor:microui"

CIRCLE_RADIUS :: f32(4.0)
CIRCLE_OMEGA :: f32(1.2)
GROUND_HALF :: [3]f32{20, 0.5, 20}
WALL_HEIGHT :: f32(2.0)
WALL_THICK :: f32(0.5)
DEG2RAD :: math.PI / 180.0

Shape :: enum {
  Sphere,
  Box,
  Cylinder,
}

Kind :: enum {
  Static,
  DynamicStill,
  DynamicCircle,
}

Body :: struct {
  kind:          Kind,
  shape:         Shape,
  size:          [3]f32, // Sphere: x=radius; Box: half extents; Cylinder: x=radius y=height
  position:      [3]f32, // for DynamicCircle this is the orbit center
  euler_deg:     [3]f32,
  mass:          mu.Real,
  node:          world.NodeHandle, // physics-driven parent
  visual:        world.NodeHandle, // mesh child; mesh swapped in place on shape change
  // physics handles (recreated on reset)
  is_static:     bool,
  static_handle: physics.StaticRigidBodyHandle,
  dyn_handle:    physics.DynamicRigidBodyHandle,
  // change tracking
  built_shape:   Shape,
  built_size:    [3]f32,
  applied_pos:   [3]f32,
  applied_rot:   quaternion128,
  applied_valid: bool,
}

SceneryBox :: struct {
  center: [3]f32,
  half:   [3]f32,
  node:   world.NodeHandle,
}

bodies: [3]Body
scenery: [5]SceneryBox // ground + 4 walls (static, not editable)
selected: int = -1
circle_angle: f32

default_bodies :: proc() -> [3]Body {
  return {
    {kind = .Static, shape = .Box, size = {1.5, 1.5, 1.5}, position = {0, 1.5, -4.5}, mass = 1},
    {kind = .DynamicStill, shape = .Sphere, size = {1.2, 1.2, 1.2}, position = {0, 1.2, 4.5}, mass = 5},
    {kind = .DynamicCircle, shape = .Sphere, size = {1.0, 1.0, 1.0}, position = {0, 1.5, 0}, mass = 3},
  }
}

main :: proc() {
  mjolnir.run_app(
    {
      title = "Collision",
      width = 1100,
      height = 760,
      debug_ui = true,
      setup = setup,
      update = update,
      pre_render = panel,
    },
  )
}

setup :: proc(engine: ^mjolnir.Engine) {
  world.main_camera_look_at(&engine.world, {16, 14, 16}, {0, 1.5, 0})
  world.spawn_light_directional(
    &engine.world,
    {-6, 14, 6},
    {1, 0.96, 0.9, 1},
    6.0,
    true,
  )
  spawn_scene(engine)
}

// One-time creation of nodes + visuals + physics bodies.
spawn_scene :: proc(engine: ^mjolnir.Engine) {
  wy := WALL_HEIGHT
  scenery[0] = {center = {0, -GROUND_HALF.y, 0}, half = GROUND_HALF}
  scenery[1] = {center = {GROUND_HALF.x, wy, 0}, half = {WALL_THICK, WALL_HEIGHT, GROUND_HALF.z}}
  scenery[2] = {center = {-GROUND_HALF.x, wy, 0}, half = {WALL_THICK, WALL_HEIGHT, GROUND_HALF.z}}
  scenery[3] = {center = {0, wy, GROUND_HALF.z}, half = {GROUND_HALF.x, WALL_HEIGHT, WALL_THICK}}
  scenery[4] = {center = {0, wy, -GROUND_HALF.z}, half = {GROUND_HALF.x, WALL_HEIGHT, WALL_THICK}}
  for &s in scenery {
    s.node = mjolnir.spawn_static(
      engine,
      s.center,
      physics.BoxCollider{half_extents = s.half},
      world.get_builtin_mesh(&engine.world, .CUBE),
      world.get_builtin_material(&engine.world, .GRAY),
    )
  }

  bodies = default_bodies()
  for &b in bodies {
    q := obj_rotation(&b)
    b.node, _ = world.spawn(&engine.world, b.position)
    world.rotate(&engine.world, b.node, q)
    b.visual, _ = world.spawn_child(
      &engine.world,
      b.node,
      attachment = world.MeshAttachment {
        handle = world.get_builtin_mesh(&engine.world, obj_mesh(b.shape)),
        material = world.get_builtin_material(&engine.world, obj_color(b.kind)),
        cast_shadow = true,
      },
    )
    s := physics.collider_visual_scale(obj_collider(&b))
    world.scale_xyz(&engine.world, b.visual, s.x, s.y, s.z)
    create_body_physics(engine, &b)
    b.built_shape = b.shape
    b.built_size = b.size
  }
}

obj_collider :: proc(b: ^Body) -> physics.Collider {
  switch b.shape {
  case .Sphere:
    return physics.SphereCollider{radius = b.size.x}
  case .Box:
    return physics.BoxCollider{half_extents = b.size}
  case .Cylinder:
    return physics.CylinderCollider{radius = b.size.x, height = b.size.y}
  }
  return physics.SphereCollider{radius = b.size.x}
}

obj_mesh :: proc(shape: Shape) -> world.Primitive {
  switch shape {
  case .Sphere:   return .SPHERE
  case .Box:      return .CUBE
  case .Cylinder: return .CYLINDER
  }
  return .SPHERE
}

obj_color :: proc(kind: Kind) -> world.Color {
  switch kind {
  case .Static:        return .BLUE
  case .DynamicStill:  return .CYAN
  case .DynamicCircle: return .MAGENTA
  }
  return .WHITE
}

obj_rotation :: proc(b: ^Body) -> quaternion128 {
  r := b.euler_deg * DEG2RAD
  qx := linalg.quaternion_angle_axis(r.x, linalg.VECTOR3F32_X_AXIS)
  qy := linalg.quaternion_angle_axis(r.y, linalg.VECTOR3F32_Y_AXIS)
  qz := linalg.quaternion_angle_axis(r.z, linalg.VECTOR3F32_Z_AXIS)
  return qy * qx * qz
}

// Create the physics body for b's current descriptor and wire the node to it.
create_body_physics :: proc(engine: ^mjolnir.Engine, b: ^Body) {
  col := obj_collider(b)
  q := obj_rotation(b)
  if b.kind == .Static {
    b.is_static = true
    b.static_handle = physics.create_static_body(&engine.physics, b.position, q, col)
    physics.rebuild_static_bvh(&engine.physics)
  } else {
    b.is_static = false
    b.dyn_handle = physics.create_dynamic_body(&engine.physics, b.position, q, f32(b.mass), col)
    if body, ok := physics.get_dynamic_body(&engine.physics, b.dyn_handle); ok {
      physics.set_inertia_from_collider(body, col)
    }
    if n, ok := world.node(&engine.world, b.node); ok {
      n.attachment = world.RigidBodyAttachment{body_handle = b.dyn_handle}
    }
  }
}

apply_shape :: proc(engine: ^mjolnir.Engine, b: ^Body) {
  col := obj_collider(b)
  if b.is_static {
    if sb, ok := physics.get_static_body(&engine.physics, b.static_handle); ok {
      sb.collider = col
      physics.update_cached_aabb(&sb.base)
    }
    physics.rebuild_static_bvh(&engine.physics)
  } else {
    if body, ok := physics.get_dynamic_body(&engine.physics, b.dyn_handle); ok {
      body.collider = col
      physics.set_inertia_from_collider(body, col)
      physics.update_cached_aabb(&body.base)
    }
    physics.rebuild_dynamic_bvh(&engine.physics)
  }
  if v, ok := world.node(&engine.world, b.visual); ok {
    v.attachment = world.MeshAttachment {
      handle = world.get_builtin_mesh(&engine.world, obj_mesh(b.shape)),
      material = world.get_builtin_material(&engine.world, obj_color(b.kind)),
      cast_shadow = true,
    }
  }
  s := physics.collider_visual_scale(col)
  world.scale_xyz(&engine.world, b.visual, s.x, s.y, s.z) // marks the node dirty → re-staged
  b.built_shape = b.shape
  b.built_size = b.size
}

drive_circle :: proc(b: ^Body, body: ^physics.DynamicRigidBody) {
  a := circle_angle
  center := b.position
  target := center + [3]f32{math.cos(a), 0, math.sin(a)} * CIRCLE_RADIUS
  tangent :=
    [3]f32{-math.sin(a), 0, math.cos(a)} * (CIRCLE_RADIUS * CIRCLE_OMEGA)
  to_target := target - body.position
  desired := tangent + to_target * 3.0
  physics.apply_impulse(body, (desired - body.velocity) * f32(b.mass))
}

update :: proc(engine: ^mjolnir.Engine, dt: f32) {
  if !engine.physics.paused do circle_angle += dt * CIRCLE_OMEGA

  for i in 0 ..< len(bodies) {
    b := &bodies[i]
    if b.shape != b.built_shape || b.size != b.built_size {
      apply_shape(engine, b)
    }
  }

  for i in 0 ..< len(bodies) {
    b := &bodies[i]
    q := obj_rotation(b)
    if b.is_static {
      if !b.applied_valid || b.position != b.applied_pos || q != b.applied_rot {
        if sb, ok := physics.get_static_body(&engine.physics, b.static_handle);
           ok {
          sb.position = b.position
          sb.rotation = q
          physics.update_cached_aabb(&sb.base)
        }
        world.translate(&engine.world, b.node, b.position)
        world.rotate(&engine.world, b.node, q)
        physics.rebuild_static_bvh(&engine.physics)
        b.applied_pos = b.position
        b.applied_rot = q
        b.applied_valid = true
      }
      continue
    }
    body, ok := physics.get_dynamic_body(&engine.physics, b.dyn_handle)
    if !ok do continue
    if b.mass > 0 do physics.set_mass(body, f32(b.mass))
    if b.kind == .DynamicCircle {
      if !engine.physics.paused do drive_circle(b, body)
    } else if !b.applied_valid ||
       b.position != b.applied_pos ||
       q != b.applied_rot {
      // DynamicStill: teleport only when the user edits the transform.
      body.position = b.position
      body.rotation = q
      body.velocity = {}
      body.angular_velocity = {}
      physics.wake_up(body)
      b.applied_pos = b.position
      b.applied_rot = q
      b.applied_valid = true
    }
  }

  handle_picking(engine)
  draw_contacts(engine, dt)
  draw_selection(engine)
}

handle_picking :: proc(engine: ^mjolnir.Engine) {
  if engine.input.consumed_by_ui do return
  if !mjolnir.is_mouse_pressed(engine, 0) do return
  cam, has_cam := world.main_camera(&engine.world)
  if !has_cam do return
  mp := mjolnir.input_mouse_position(&engine.input)
  origin, dir := world.camera_viewport_to_world_ray(
    cam,
    f32(mp.x),
    f32(mp.y),
  )
  hit := physics.raycast(
    &engine.physics,
    geometry.Ray{origin = origin, direction = dir},
  )
  if !hit.hit {
    selected = -1
    return
  }
  selected = -1
  #partial switch h in hit.body_handle {
  case physics.DynamicRigidBodyHandle:
    for &b, i in bodies do if !b.is_static && b.dyn_handle == h do selected = i
  case physics.StaticRigidBodyHandle:
    for &b, i in bodies do if b.is_static && b.static_handle == h do selected = i
  }
}

draw_contacts :: proc(engine: ^mjolnir.Engine, dt: f32) {
  life := max(dt * 2.0, 0.04)
  draw_one :: proc(
    engine: ^mjolnir.Engine,
    p, n: [3]f32,
    penetration, normal_impulse, dt, life: f32,
  ) {
    // hit point
    mjolnir.debug_sphere(engine, p, 0.12, {1, 0.9, 0, 1}, life, true)
    // contact normal (cyan, fixed length)
    mjolnir.debug_arrow(engine, p, p + n * 0.9, {0, 1, 1, 1}, life, true)
    // penetration depth (magenta, into surface)
    if penetration > 0 {
      mjolnir.debug_segment(
        engine,
        p,
        p - n * penetration,
        {1, 0, 1, 1},
        life,
        true,
      )
    }
    // approximate contact force = impulse / dt (red, scaled + clamped)
    if dt > 0 && normal_impulse > 0 {
      force := normal_impulse / dt
      flen := clamp(force * 0.01, 0.1, 4.0)
      mjolnir.debug_arrow(engine, p, p + n * flen, {1, 0.25, 0.2, 1}, life, true)
    }
  }
  for c in engine.physics.static_contacts {
    for i in 0 ..< c.count {
      p := c.points[i]
      draw_one(engine, p.point, c.normal, p.penetration, p.normal_impulse, dt, life)
    }
  }
  for c in engine.physics.dynamic_contacts {
    for i in 0 ..< c.count {
      p := c.points[i]
      draw_one(engine, p.point, c.normal, p.penetration, p.normal_impulse, dt, life)
    }
  }
}

body_transform :: proc(
  engine: ^mjolnir.Engine,
  b: ^Body,
) -> (
  pos: [3]f32,
  rot: quaternion128,
) {
  if b.is_static {
    return b.position, obj_rotation(b)
  }
  if body, ok := physics.get_dynamic_body(&engine.physics, b.dyn_handle); ok {
    return body.position, body.rotation
  }
  return b.position, obj_rotation(b)
}

draw_selection :: proc(engine: ^mjolnir.Engine) {
  if selected < 0 do return
  b := &bodies[selected]
  pos, rot := body_transform(engine, b)
  mjolnir.debug_axes(engine, pos, rot, 1.8, 0, true)
  green := [4]f32{0.2, 1, 0.3, 1}
  switch b.shape {
  case .Sphere:
    mjolnir.debug_sphere(engine, pos, b.size.x, green, 0, true)
  case .Box:
    mjolnir.debug_cube(engine, pos, rot, b.size * 2, green, 0, true)
  case .Cylinder:
    axis := linalg.quaternion_mul_vector3(rot, [3]f32{0, 1, 0})
    h := b.size.y * 0.5
    mjolnir.debug_circle(engine, pos + axis * h, axis, b.size.x, green, 0, true)
    mjolnir.debug_circle(engine, pos - axis * h, axis, b.size.x, green, 0, true)
  }
}

panel :: proc(engine: ^mjolnir.Engine) {
  ctx := mjolnir.ui_ctx(engine)
  if mu.window(ctx, "Inspector", {20, 220, 320, 500}, {.NO_CLOSE}) {
    if selected <= 0 {
      mu.label(ctx, "Left-click an object to select it.")
    } else {
      b := &bodies[selected]
      mu.layout_row(ctx, {-1}, 0)
      mu.layout_row(ctx, {90, 90, 90}, 0)
      if .SUBMIT in mu.button(ctx, "Sphere") do b.shape = .Sphere
      if .SUBMIT in mu.button(ctx, "Box") do b.shape = .Box
      if .SUBMIT in mu.button(ctx, "Cylinder") do b.shape = .Cylinder

      mu.layout_row(ctx, {-1}, 0)
      switch b.shape {
      case .Sphere:
        mu.label(ctx, "Radius")
        mu.slider(ctx, &b.size.x, 0.2, 4.0)
      case .Box:
        mu.label(ctx, "Half extents X / Y / Z")
        mu.slider(ctx, &b.size.x, 0.2, 4.0)
        mu.slider(ctx, &b.size.y, 0.2, 4.0)
        mu.slider(ctx, &b.size.z, 0.2, 4.0)
      case .Cylinder:
        mu.label(ctx, "Radius / Height")
        mu.slider(ctx, &b.size.x, 0.2, 4.0)
        mu.slider(ctx, &b.size.y, 0.4, 6.0)
      }

      label := b.kind == .DynamicCircle ? "Orbit center X / Y / Z" : "Position X / Y / Z"
      mu.label(ctx, label)
      mu.slider(ctx, &b.position.x, -12, 12)
      mu.slider(ctx, &b.position.y, 0, 12)
      mu.slider(ctx, &b.position.z, -12, 12)

      mu.label(ctx, "Rotation X / Y / Z (deg)")
      mu.slider(ctx, &b.euler_deg.x, -180, 180)
      mu.slider(ctx, &b.euler_deg.y, -180, 180)
      mu.slider(ctx, &b.euler_deg.z, -180, 180)

      if !b.is_static {
        mu.label(ctx, "Mass")
        mu.slider(ctx, &b.mass, 0.5, 50)
      }
    }
  }
  if mu.window(ctx, "Collision", {20, 20, 320, 180}, {.NO_CLOSE}) {
    mu.layout_row(ctx, {150, -1}, 0)
    if .SUBMIT in mu.button(ctx, "Reset Scene") {
      mjolnir.schedule_teardown(engine)
      mjolnir.schedule_setup(engine)
    }
    if .SUBMIT in mu.button(ctx, engine.physics.paused ? "Resume" : "Pause") {
      engine.physics.paused = !engine.physics.paused
    }

    perf := engine.physics.last_perf
    mu.label(
      ctx,
      fmt.tprintf(
        "Contacts: %d static, %d dynamic",
        perf.static_contact_count,
        perf.dynamic_contact_count,
      ),
    )


    mu.layout_row(ctx, {-1}, 0)
    shown := 0
    for c in engine.physics.static_contacts {
      if shown >= 6 do break
      shown += 1
      p := c.points[0]
      mu.label(
        ctx,
        fmt.tprintf(
          "Static Contact x%d (%.1f,%.1f,%.1f) pen %.2f J %.1f",
          c.count, p.point.x, p.point.y, p.point.z, p.penetration, p.normal_impulse,
        ),
      )
    }
    for c in engine.physics.dynamic_contacts {
      if shown >= 6 do break
      shown += 1
      p := c.points[0]
      mu.label(
        ctx,
        fmt.tprintf(
          "Dynamic Contact x%d (%.1f,%.1f,%.1f) pen %.2f J %.1f",
          c.count, p.point.x, p.point.y, p.point.z, p.penetration, p.normal_impulse,
        ),
      )
    }
  }
}
