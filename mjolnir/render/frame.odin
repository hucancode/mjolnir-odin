package render

import geom "../geometry"
import "../gpu"
import "ambient"
import "debug_line"
import "debug_ui"
import depth_pyramid_system "depth_pyramid"
import "direct_light"
import "geometry"
import "line_strip"
import "occlusion_culling"
import particles_compute "particles_compute"
import particles_render "particles_render"
import "post_process"
import "random_color"
import shadow_culling_system "shadow_culling"
import shadow_render_system "shadow_render"
import shadow_sphere_culling_system "shadow_sphere_culling"
import shadow_sphere_render_system "shadow_sphere_render"
import "sprite"
import "transparent"
import ui_render "ui"
import vk "vendor:vulkan"
import "wireframe"

@(private)
record_compute_commands :: proc(
  self: ^Manager,
  frame_index: u32,
  gctx: ^gpu.GPUContext,
  cull_camera_indices: []u32,
) -> vk.Result {
  cmd :=
    gctx.has_async_compute ? self.internal.compute_command_buffers[frame_index] : self.internal.command_buffers[frame_index]
  // Compute for frame N prepares data for frame N+1
  // Buffer indices with FRAMES_IN_FLIGHT=2: frame N uses buffer [N], produces data for buffer [N+1]
  next_frame_index := frame_next(frame_index)
  for cam_index in cull_camera_indices {
    cam, ok := &self.cameras[cam_index]
    if !ok do continue
    depth_pyramid_system.build_pyramid(
      &self.internal.depth_pyramid,
      cmd,
      self.node_count,
      &cam.depth_pyramid[frame_index],
      cam.depth_reduce_descriptor_sets[frame_index][:],
    ) // Build pyramid[N]
    prev_frame := frame_prev(next_frame_index)
    occlusion_culling.perform_culling(
      &self.internal.visibility,
      cmd,
      cam_index,
      next_frame_index,
      self.node_count,
      &cam.draws,
      cam.descriptor_set[next_frame_index],
      cam.depth_pyramid[prev_frame].width,
      cam.depth_pyramid[prev_frame].height,
    ) // Write draw_list[N+1]
  }
  particles_compute.simulate(
    &self.internal.particles_compute,
    cmd,
    self.internal.node_data_buffer.descriptor_set,
  )
  return .SUCCESS
}

@(private)
render_shadow_depth :: proc(
  self: ^Manager,
  frame_index: u32,
  cache: map[u32]shadow_render_system.ShadowFrameInfo,
) -> vk.Result {
  cmd := self.internal.command_buffers[frame_index]
  light_node_indices := sorted_light_indices(self)
  for light_node_index in light_node_indices {
    light := self.internal.lights[light_node_index]
    info := cache[light_node_index]
    switch variant in light {
    case SpotLight, DirectionalLight:
      shadow, has_shadow := &self.internal.shadow_maps[light_node_index]
      if !has_shadow do continue
      frustum_planes := geom.make_frustum(info.view_projection).planes
      shadow_culling_system.execute(
        &self.internal.shadow_culling,
        cmd,
        self.node_count,
        frustum_planes,
        shadow.draw_count[frame_index].buffer,
        shadow.descriptor_sets[frame_index],
      )
      shadow_render_system.render(
        &self.internal.shadow_render,
        cmd,
        &self.texture_manager,
        info.view_projection,
        shadow.shadow_map_2d[frame_index],
        shadow.draw_commands[frame_index],
        shadow.draw_count[frame_index],
        self.texture_manager.descriptor_set,
        self.internal.bone_buffer.descriptor_sets[frame_index],
        self.internal.material_buffer.descriptor_set,
        self.internal.node_data_buffer.descriptor_set,
        self.internal.mesh_data_buffer.descriptor_set,
        self.mesh_manager.vertex_skinning_buffer.descriptor_set,
        self.mesh_manager.vertex_buffer.buffer,
        self.mesh_manager.index_buffer.buffer,
        frame_index,
      )
    case PointLight:
      shadow, has_shadow := &self.internal.shadow_map_cubes[light_node_index]
      if !has_shadow do continue
      shadow_sphere_culling_system.execute(
        &self.internal.shadow_sphere_culling,
        cmd,
        self.node_count,
        variant.position,
        variant.radius,
        shadow.draw_count[frame_index].buffer,
        shadow.descriptor_sets[frame_index],
      )
      shadow_sphere_render_system.render(
        &self.internal.shadow_sphere_render,
        cmd,
        &self.texture_manager,
        info.projection,
        info.near,
        info.far,
        variant.position,
        shadow.shadow_map_cube[frame_index],
        shadow.draw_commands[frame_index],
        shadow.draw_count[frame_index],
        self.texture_manager.descriptor_set,
        self.internal.bone_buffer.descriptor_sets[frame_index],
        self.internal.material_buffer.descriptor_set,
        self.internal.node_data_buffer.descriptor_set,
        self.internal.mesh_data_buffer.descriptor_set,
        self.mesh_manager.vertex_skinning_buffer.descriptor_set,
        self.mesh_manager.vertex_buffer.buffer,
        self.mesh_manager.index_buffer.buffer,
        frame_index,
      )
    }
  }
  return .SUCCESS
}

@(private)
record_geometry_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  cam_index: u32,
  cam: ^CameraTarget,
  enabled_passes: PassTypeSet,
) -> vk.Result {
  if .GEOMETRY not_in enabled_passes do return .SUCCESS
  geometry.record(
    &self.internal.geometry,
    cam_index,
    self.internal.command_buffers[frame_index],
    &self.texture_manager,
    cam.attachments[.POSITION][frame_index],
    cam.attachments[.NORMAL][frame_index],
    cam.attachments[.ALBEDO][frame_index],
    cam.attachments[.METALLIC_ROUGHNESS][frame_index],
    cam.attachments[.EMISSIVE][frame_index],
    cam.attachments[.FINAL_IMAGE][frame_index],
    cam.attachments[.DEPTH][frame_index],
    self.internal.camera_buffer.descriptor_sets[frame_index],
    self.texture_manager.descriptor_set,
    self.internal.bone_buffer.descriptor_sets[frame_index],
    self.internal.material_buffer.descriptor_set,
    self.internal.node_data_buffer.descriptor_set,
    self.internal.mesh_data_buffer.descriptor_set,
    self.mesh_manager.vertex_skinning_buffer.descriptor_set,
    self.mesh_manager.vertex_buffer.buffer,
    self.mesh_manager.index_buffer.buffer,
    cam.draws[.OPAQUE].commands[frame_index].buffer,
    cam.draws[.OPAQUE].count[frame_index].buffer,
    MAX_NODES_IN_SCENE,
  )
  return .SUCCESS
}

@(private)
record_lighting_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  cam_index: u32,
  cam: ^CameraTarget,
  enabled_passes: PassTypeSet,
  cache: map[u32]shadow_render_system.ShadowFrameInfo,
) -> vk.Result {
  if .LIGHTING not_in enabled_passes do return .SUCCESS
  cmd := self.internal.command_buffers[frame_index]
  ambient.record(
    &self.ambient,
    cam_index,
    cmd,
    &self.texture_manager,
    cam.attachments[.FINAL_IMAGE][frame_index],
    self.internal.camera_buffer.descriptor_sets[frame_index],
    cam.attachments[.POSITION][frame_index].index,
    cam.attachments[.NORMAL][frame_index].index,
    cam.attachments[.ALBEDO][frame_index].index,
    cam.attachments[.METALLIC_ROUGHNESS][frame_index].index,
    cam.attachments[.EMISSIVE][frame_index].index,
  )
  direct_light.begin_pass(
    &self.internal.direct_light,
    cam.attachments[.FINAL_IMAGE][frame_index],
    cam.attachments[.DEPTH][frame_index],
    &self.texture_manager,
    cmd,
    self.internal.camera_buffer.descriptor_sets[frame_index],
  )
  light_node_indices := sorted_light_indices(self)
  for light_node_index in light_node_indices {
    light := self.internal.lights[light_node_index]
    info := cache[light_node_index]
    switch variant in light {
    case PointLight:
      shadow_map_idx: u32 = 0xFFFFFFFF
      shadow_view_projection := matrix[4, 4]f32{}
      if sm, ok := self.internal.shadow_map_cubes[light_node_index]; ok {
        shadow_map_idx = sm.shadow_map_cube[frame_index].index
        shadow_view_projection = info.projection
      }
      direct_light.render_point_light(
        &self.internal.direct_light,
        cam_index,
        cam.attachments[.POSITION][frame_index].index,
        cam.attachments[.NORMAL][frame_index].index,
        cam.attachments[.ALBEDO][frame_index].index,
        cam.attachments[.METALLIC_ROUGHNESS][frame_index].index,
        variant.color,
        variant.position,
        variant.radius,
        shadow_map_idx,
        shadow_view_projection,
        cmd,
      )
    case SpotLight:
      shadow_map_idx: u32 = 0xFFFFFFFF
      shadow_view_projection := matrix[4, 4]f32{}
      if sm, ok := self.internal.shadow_maps[light_node_index]; ok {
        shadow_map_idx = sm.shadow_map_2d[frame_index].index
        shadow_view_projection = info.view_projection
      }
      direct_light.render_spot_light(
        &self.internal.direct_light,
        cam_index,
        cam.attachments[.POSITION][frame_index].index,
        cam.attachments[.NORMAL][frame_index].index,
        cam.attachments[.ALBEDO][frame_index].index,
        cam.attachments[.METALLIC_ROUGHNESS][frame_index].index,
        variant.color,
        variant.position,
        variant.direction,
        variant.radius,
        variant.angle_inner,
        variant.angle_outer,
        shadow_map_idx,
        shadow_view_projection,
        cmd,
      )
    case DirectionalLight:
      shadow_map_idx: u32 = 0xFFFFFFFF
      shadow_view_projection := matrix[4, 4]f32{}
      if sm, ok := self.internal.shadow_maps[light_node_index]; ok {
        shadow_map_idx = sm.shadow_map_2d[frame_index].index
        shadow_view_projection = info.view_projection
      }
      direct_light.render_directional_light(
        &self.internal.direct_light,
        cam_index,
        cam.attachments[.POSITION][frame_index].index,
        cam.attachments[.NORMAL][frame_index].index,
        cam.attachments[.ALBEDO][frame_index].index,
        cam.attachments[.METALLIC_ROUGHNESS][frame_index].index,
        variant.color,
        variant.direction,
        shadow_map_idx,
        shadow_view_projection,
        cmd,
      )
    }
  }
  direct_light.end_pass(cmd)
  return .SUCCESS
}

@(private)
record_particles_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  cam_index: u32,
  cam: ^CameraTarget,
  enabled_passes: PassTypeSet,
) -> vk.Result {
  if .PARTICLES not_in enabled_passes do return .SUCCESS
  particles_render.record(
    &self.internal.particles_render,
    self.internal.command_buffers[frame_index],
    cam_index,
    cam.attachments[.FINAL_IMAGE][frame_index],
    cam.attachments[.DEPTH][frame_index],
    &self.texture_manager,
    self.internal.camera_buffer.descriptor_sets[frame_index],
    self.texture_manager.descriptor_set,
    self.internal.particles_compute.compact_particle_buffer.buffer,
    self.internal.particles_compute.draw_command_buffer.buffer,
  )
  return .SUCCESS
}

@(private)
record_transparency_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  gctx: ^gpu.GPUContext,
  cam_index: u32,
  cam: ^CameraTarget,
  enabled_passes: PassTypeSet,
) -> vk.Result {
  if .TRANSPARENCY not_in enabled_passes do return .SUCCESS
  cmd := self.internal.command_buffers[frame_index]

  // Open a single render scope shared by all 5 sub-passes (transparent /
  // wireframe / random_color / line_strip / sprite). Hoist all draw-buffer
  // barriers + depth layout transition here — Vulkan disallows
  // vkCmdPipelineBarrier inside a dynamic rendering instance.
  color_texture := gpu.get_texture_2d(
    &self.texture_manager,
    cam.attachments[.FINAL_IMAGE][frame_index],
  )
  depth_texture := gpu.get_texture_2d(
    &self.texture_manager,
    cam.attachments[.DEPTH][frame_index],
  )
  for pipe in ([?]DrawPipeline {
       .TRANSPARENT,
       .WIREFRAME,
       .RANDOM_COLOR,
       .LINE_STRIP,
       .SPRITE,
     }) {
    if pipe == .WIREFRAME && .WIREFRAME not_in enabled_passes do continue
    if pipe == .RANDOM_COLOR && .RANDOM_COLOR not_in enabled_passes do continue
    if pipe == .LINE_STRIP && .LINE_STRIP not_in enabled_passes do continue
    if pipe == .SPRITE && .SPRITE not_in enabled_passes do continue
    cmds := &cam.draws[pipe].commands[frame_index]
    cnt := &cam.draws[pipe].count[frame_index]
    gpu.buffer_barrier(
      cmd,
      cmds.buffer,
      vk.DeviceSize(cmds.bytes_count),
      {.SHADER_WRITE},
      {.INDIRECT_COMMAND_READ},
      {.COMPUTE_SHADER},
      {.DRAW_INDIRECT},
    )
    gpu.buffer_barrier(
      cmd,
      cnt.buffer,
      vk.DeviceSize(cnt.bytes_count),
      {.SHADER_WRITE},
      {.INDIRECT_COMMAND_READ},
      {.COMPUTE_SHADER},
      {.DRAW_INDIRECT},
    )
  }
  gpu.memory_barrier(
    cmd,
    {.SHADER_READ},
    {.DEPTH_STENCIL_ATTACHMENT_READ, .DEPTH_STENCIL_ATTACHMENT_WRITE},
    {.FRAGMENT_SHADER},
    {.EARLY_FRAGMENT_TESTS},
  )
  gpu.begin_rendering(
    cmd,
    depth_texture.spec.extent,
    gpu.create_depth_attachment(depth_texture, .LOAD, .STORE),
    gpu.create_color_attachment(color_texture, .LOAD, .STORE),
  )
  gpu.set_viewport_scissor(cmd, depth_texture.spec.extent)

  camera_set := self.internal.camera_buffer.descriptor_sets[frame_index]
  textures_set := self.texture_manager.descriptor_set
  bone_set := self.internal.bone_buffer.descriptor_sets[frame_index]
  material_set := self.internal.material_buffer.descriptor_set
  node_data_set := self.internal.node_data_buffer.descriptor_set
  mesh_data_set := self.internal.mesh_data_buffer.descriptor_set
  skinning_set := self.mesh_manager.vertex_skinning_buffer.descriptor_set
  vbuf := self.mesh_manager.vertex_buffer.buffer
  ibuf := self.mesh_manager.index_buffer.buffer

  transparent.record(
    &self.internal.transparent_renderer,
    cmd, cam_index,
    camera_set, textures_set, bone_set, material_set,
    node_data_set, mesh_data_set, skinning_set,
    vbuf, ibuf,
    &cam.draws[.TRANSPARENT].commands[frame_index],
    &cam.draws[.TRANSPARENT].count[frame_index],
    MAX_NODES_IN_SCENE,
  )
  if .WIREFRAME in enabled_passes {
    wireframe.record(
      &self.internal.wireframe_renderer,
      cmd, cam_index,
      camera_set, textures_set, bone_set, material_set,
      node_data_set, mesh_data_set, skinning_set,
      vbuf, ibuf,
      &cam.draws[.WIREFRAME].commands[frame_index],
      &cam.draws[.WIREFRAME].count[frame_index],
      MAX_NODES_IN_SCENE,
    )
  }
  if .RANDOM_COLOR in enabled_passes {
    random_color.record(
      &self.internal.random_color_renderer,
      cmd, cam_index,
      camera_set, textures_set, bone_set, material_set,
      node_data_set, mesh_data_set, skinning_set,
      vbuf, ibuf,
      &cam.draws[.RANDOM_COLOR].commands[frame_index],
      &cam.draws[.RANDOM_COLOR].count[frame_index],
      MAX_NODES_IN_SCENE,
    )
  }
  if .LINE_STRIP in enabled_passes {
    line_strip.record(
      &self.internal.line_strip_renderer,
      cmd, cam_index,
      camera_set, textures_set, bone_set, material_set,
      node_data_set, mesh_data_set, skinning_set,
      vbuf, ibuf,
      &cam.draws[.LINE_STRIP].commands[frame_index],
      &cam.draws[.LINE_STRIP].count[frame_index],
      MAX_NODES_IN_SCENE,
    )
  }
  if .SPRITE in enabled_passes {
    sprite.record(
      &self.internal.sprite_renderer,
      cmd, cam_index,
      camera_set, textures_set, node_data_set,
      self.internal.sprite_buffer.descriptor_set,
      vbuf, ibuf,
      &cam.draws[.SPRITE].commands[frame_index],
      &cam.draws[.SPRITE].count[frame_index],
      MAX_NODES_IN_SCENE,
    )
  }

  vk.CmdEndRendering(cmd)
  // Make depth writes visible to shader sampling (e.g. depth pyramid build
  // at start of next frame).
  gpu.memory_barrier(
    cmd,
    {.DEPTH_STENCIL_ATTACHMENT_WRITE},
    {.SHADER_READ},
    {.LATE_FRAGMENT_TESTS},
    {.COMPUTE_SHADER, .FRAGMENT_SHADER},
  )
  return .SUCCESS
}

@(private)
record_debug_line_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  cam_index: u32,
  cam: ^CameraTarget,
  swapchain_view: vk.ImageView,
  swapchain_extent: vk.Extent2D,
  enabled_passes: PassTypeSet,
) {
  if .DEBUG_LINE not_in enabled_passes do return
  depth := gpu.get_texture_2d(
    &self.texture_manager,
    cam.attachments[.DEPTH][frame_index],
  )
  if depth == nil do return
  debug_line.record(
    &self.internal.debug_line_renderer,
    self.internal.command_buffers[frame_index],
    frame_index,
    self.internal.camera_buffer.descriptor_sets[frame_index],
    cam_index,
    swapchain_view,
    swapchain_extent,
    depth,
  )
}

@(private)
record_post_process_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  cam: ^CameraTarget,
  swapchain_extent: vk.Extent2D,
  swapchain_image: vk.Image,
  swapchain_view: vk.ImageView,
  enabled_passes: PassTypeSet,
) -> vk.Result {
  if .POST_PROCESS not_in enabled_passes do return .SUCCESS
  cmd := self.internal.command_buffers[frame_index]
  if final_image := gpu.get_texture_2d(
    &self.texture_manager,
    cam.attachments[.FINAL_IMAGE][frame_index],
  ); final_image != nil {
    gpu.memory_barrier(
      cmd,
      {.COLOR_ATTACHMENT_WRITE},
      {.SHADER_READ},
      {.COLOR_ATTACHMENT_OUTPUT},
      {.FRAGMENT_SHADER},
    )
  }
  gpu.image_discard_barrier(
    cmd,
    swapchain_image,
    {.COLOR_ATTACHMENT_WRITE},
    {.COLOR_ATTACHMENT_OUTPUT},
    {.COLOR},
  )
  post_process.record(
    &self.post_process,
    cmd,
    swapchain_extent,
    swapchain_view,
    cam.attachments[.FINAL_IMAGE][frame_index].index,
    cam.attachments[.POSITION][frame_index].index,
    cam.attachments[.NORMAL][frame_index].index,
    cam.attachments[.ALBEDO][frame_index].index,
    cam.attachments[.METALLIC_ROUGHNESS][frame_index].index,
    cam.attachments[.EMISSIVE][frame_index].index,
    cam.attachments[.DEPTH][frame_index].index,
    &self.texture_manager,
  )
  return .SUCCESS
}

@(private)
record_ui_pass :: proc(
  self: ^Manager,
  frame_index: u32,
  gctx: ^gpu.GPUContext,
  swapchain_view: vk.ImageView,
  swapchain_extent: vk.Extent2D,
  enabled_passes: PassTypeSet,
) {
  if .UI not_in enabled_passes do return
  cmd := self.internal.command_buffers[frame_index]
  // UI rendering pass - renders on top of post-processed image
  rendering_attachment_info := vk.RenderingAttachmentInfo {
    sType       = .RENDERING_ATTACHMENT_INFO,
    imageView   = swapchain_view,
    imageLayout = .GENERAL,
    loadOp      = .LOAD,
    storeOp     = .STORE,
  }

  rendering_info := vk.RenderingInfo {
    sType = .RENDERING_INFO,
    renderArea = {extent = swapchain_extent},
    layerCount = 1,
    colorAttachmentCount = 1,
    pColorAttachments = &rendering_attachment_info,
  }

  vk.CmdBeginRendering(cmd, &rendering_info)

  // Set viewport and scissor
  viewport := vk.Viewport {
    x        = 0,
    y        = f32(swapchain_extent.height),
    width    = f32(swapchain_extent.width),
    height   = -f32(swapchain_extent.height),
    minDepth = 0.0,
    maxDepth = 1.0,
  }
  scissor := vk.Rect2D {
    offset = {0, 0},
    extent = swapchain_extent,
  }
  vk.CmdSetViewport(cmd, 0, 1, &viewport)
  vk.CmdSetScissor(cmd, 0, 1, &scissor)

  ui_render.render(
    &self.internal.ui,
    gctx,
    &self.texture_manager,
    cmd,
    swapchain_extent.width,
    swapchain_extent.height,
    frame_index,
  )

  vk.CmdEndRendering(cmd)
}

// record_frame drives the entire per-frame command sequence: shadow maps,
// per-camera passes (geometry, lighting, particles, transparency), debug,
// post-process, UI, async compute, optional debug-UI overlay, and the
// final swapchain transition to PRESENT_SRC — the one layout transition
// unified image layouts doesn't cover (the compositor is outside Vulkan).
record_frame :: proc(
  self: ^Manager,
  gctx: ^gpu.GPUContext,
  frame_index: u32,
  swapchain_image: vk.Image,
  swapchain_view: vk.ImageView,
  swapchain_extent: vk.Extent2D,
  main_camera_index: u32,
  active_camera_indices: []u32,
  debug_ui_enabled: bool,
) -> vk.Result {
  cmd := self.internal.command_buffers[frame_index]
  gpu.begin_record(cmd) or_return
  shadow_cache := build_shadow_cache(self)
  render_shadow_depth(self, frame_index, shadow_cache) or_return

  cull_indices := make(
    [dynamic]u32,
    0,
    len(active_camera_indices),
    context.temp_allocator,
  )
  defer delete(cull_indices)
  for index in active_camera_indices {
    cam, ok := &self.cameras[index]
    if !ok do continue
    if cam.enable_culling do append(&cull_indices, index)
    record_geometry_pass(self, frame_index, index, cam, cam.enabled_passes) or_return
    record_lighting_pass(self, frame_index, index, cam, cam.enabled_passes, shadow_cache) or_return
    record_particles_pass(self, frame_index, index, cam, cam.enabled_passes) or_return
    record_transparency_pass(self, frame_index, gctx, index, cam, cam.enabled_passes) or_return
  }

  main_camera_passes: PassTypeSet
  if main_cam, ok := &self.cameras[main_camera_index]; ok {
    main_camera_passes = main_cam.enabled_passes
    record_post_process_pass(
      self,
      frame_index,
      main_cam,
      swapchain_extent,
      swapchain_image,
      swapchain_view,
      main_camera_passes,
    ) or_return
    record_debug_line_pass(
      self,
      frame_index,
      main_camera_index,
      main_cam,
      swapchain_view,
      swapchain_extent,
      main_camera_passes,
    )
  }
  record_ui_pass(self, frame_index, gctx, swapchain_view, swapchain_extent, main_camera_passes)

  // Compute (potentially on a separate async queue command buffer).
  compute_cmd := cmd
  if gctx.has_async_compute {
    compute_cmd = self.internal.compute_command_buffers[frame_index]
    gpu.begin_record(compute_cmd) or_return
  }
  record_compute_commands(self, frame_index, gctx, cull_indices[:]) or_return
  if gctx.has_async_compute {
    gpu.end_record(compute_cmd) or_return
  }

  if debug_ui_enabled {
    debug_ui.record(
      &self.debug_ui,
      cmd,
      swapchain_view,
      swapchain_extent,
      self.texture_manager.descriptor_set,
    )
  }

  present_barrier := vk.ImageMemoryBarrier2 {
    sType = .IMAGE_MEMORY_BARRIER_2,
    srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
    srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
    dstStageMask = {.BOTTOM_OF_PIPE},
    oldLayout = .GENERAL,
    newLayout = .PRESENT_SRC_KHR,
    srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
    dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
    image = swapchain_image,
    subresourceRange = {aspectMask = {.COLOR}, levelCount = 1, layerCount = 1},
  }
  present_dep := vk.DependencyInfo {
    sType                   = .DEPENDENCY_INFO,
    imageMemoryBarrierCount = 1,
    pImageMemoryBarriers    = &present_barrier,
  }
  vk.CmdPipelineBarrier2(cmd, &present_dep)
  gpu.end_record(cmd) or_return
  return .SUCCESS
}
