package direct_light

import "../../geometry"
import "../../gpu"
import "../shared"
import "core:log"
import "core:math/linalg"
import vk "vendor:vulkan"

SHADER_POINT_VERT := #load("../../shader/lighting_point/vert.spv")
SHADER_POINT_FRAG := #load("../../shader/lighting_point/frag.spv")
SHADER_SPOT_VERT := #load("../../shader/lighting_spot/vert.spv")
SHADER_SPOT_FRAG := #load("../../shader/lighting_spot/frag.spv")
SHADER_DIRECTIONAL_VERT := #load("../../shader/lighting_directional/vert.spv")
SHADER_DIRECTIONAL_FRAG := #load("../../shader/lighting_directional/frag.spv")

BG_BLUE_GRAY :: [4]f32{0.0117, 0.0117, 0.0179, 1.0}
BG_DARK_GRAY :: [4]f32{0.0117, 0.0117, 0.0117, 1.0}
BG_ORANGE_GRAY :: [4]f32{0.0179, 0.0179, 0.0117, 1.0}

PointLightPushConstants :: struct {
  shadow_view_projection: matrix[4, 4]f32, // 64 bytes
  light_color:            [4]f32, // 16 bytes
  position:               [3]f32, // 12 bytes
  radius:                 f32,    // 4 bytes
  shadow_map_idx:         u32,    // 4 bytes
  scene_camera_idx:       u32,    // 4 bytes
  position_texture_index: u32,
  normal_texture_index:   u32,
  albedo_texture_index:   u32,
  metallic_texture_index: u32,
}

SpotLightPushConstants :: struct {
  shadow_view_projection:       matrix[4, 4]f32, // 64 bytes
  light_color:                  [4]f32,          // 16 bytes
  position:                     [3]f32,          // 12 bytes
  angle_inner:                  f32,             // 4 bytes
  direction:                    [3]f32,          // 12 bytes
  radius:                       f32,             // 4 bytes
  angle_outer:                  f32,             // 4 bytes
  shadow_and_camera_indices:    u32,             // 4 bytes (shadow_map_idx | scene_camera_idx << 16)
  position_and_normal_indices:  u32,             // 4 bytes (position_texture_index | normal_texture_index << 16)
  albedo_and_metallic_indices:  u32,             // 4 bytes (albedo_texture_index | metallic_texture_index << 16)
}
// Total: 128 bytes (optimized from 136 bytes)

DirectionalLightPushConstants :: struct {
  shadow_view_projection: matrix[4, 4]f32, // 64 bytes
  light_color:            [4]f32, // 16 bytes
  direction:              [3]f32, // 12 bytes
  shadow_map_idx:         u32,    // 4 bytes
  scene_camera_idx:       u32,    // 4 bytes
  position_texture_index: u32,
  normal_texture_index:   u32,
  albedo_texture_index:   u32,
  metallic_texture_index: u32,
}

Renderer :: struct {
  point_pipeline:       vk.Pipeline,
  spot_pipeline:        vk.Pipeline,
  directional_pipeline: vk.Pipeline,
  pipeline_layout:      vk.PipelineLayout,
  sphere_mesh:          LightVolumeMesh,
  cone_mesh:            LightVolumeMesh,
  triangle_mesh:        LightVolumeMesh,
}

LightVolumeMesh :: struct {
  vertex_buffer: gpu.ImmutableBuffer(geometry.Vertex),
  index_buffer:  gpu.ImmutableBuffer(u32),
  index_count:   u32,
}

@(private)
create_light_volume_mesh :: proc(
  gctx: ^gpu.GPUContext,
  out_mesh: ^LightVolumeMesh,
  geom: geometry.Geometry,
) -> vk.Result {
  out_mesh^ = {}
  vertex_buffer, ret_vertex := gpu.malloc_buffer(
    gctx,
    geometry.Vertex,
    len(geom.vertices),
    {.VERTEX_BUFFER},
  )
  if ret_vertex != .SUCCESS do return ret_vertex
  out_mesh.vertex_buffer = vertex_buffer
  index_buffer, ret_index := gpu.malloc_buffer(
    gctx,
    u32,
    len(geom.indices),
    {.INDEX_BUFFER},
  )
  if ret_index != .SUCCESS {
    gpu.buffer_destroy(gctx.device, &out_mesh.vertex_buffer)
    return ret_index
  }
  out_mesh.index_buffer = index_buffer
  if gpu.write(gctx, &out_mesh.vertex_buffer, geom.vertices) != .SUCCESS {
    destroy_light_volume_mesh(gctx.device, out_mesh)
    return .ERROR_UNKNOWN
  }
  if gpu.write(gctx, &out_mesh.index_buffer, geom.indices) != .SUCCESS {
    destroy_light_volume_mesh(gctx.device, out_mesh)
    return .ERROR_UNKNOWN
  }
  out_mesh.index_count = u32(len(geom.indices))
  return .SUCCESS
}

@(private)
destroy_light_volume_mesh :: proc(device: vk.Device, mesh: ^LightVolumeMesh) {
  gpu.buffer_destroy(device, &mesh.vertex_buffer)
  gpu.buffer_destroy(device, &mesh.index_buffer)
  mesh.index_count = 0
}

init :: proc(
  self: ^Renderer,
  gctx: ^gpu.GPUContext,
  camera_set_layout: vk.DescriptorSetLayout,
  textures_set_layout: vk.DescriptorSetLayout,
) -> (
  ret: vk.Result,
) {
  log.debug("Direct lighting renderer init")
  // Push constant size must fit largest variant (SpotLightPushConstants = 128 bytes)
  log.debugf("SpotLightPushConstants size: %d bytes", size_of(SpotLightPushConstants))
  self.pipeline_layout = gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange {
      stageFlags = {.VERTEX, .FRAGMENT},
      size = size_of(SpotLightPushConstants),
    },
    camera_set_layout,
    textures_set_layout,
  ) or_return
  defer if ret != .SUCCESS {
    vk.DestroyPipelineLayout(gctx.device, self.pipeline_layout, nil)
  }

  dynamic_states := [?]vk.DynamicState {
    .VIEWPORT,
    .SCISSOR,
    .DEPTH_COMPARE_OP,
    .CULL_MODE,
  }
  dynamic_state := gpu.create_dynamic_state(dynamic_states[:])
  vertex_input := vk.PipelineVertexInputStateCreateInfo {
    sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    vertexBindingDescriptionCount   = 1,
    pVertexBindingDescriptions      = &geometry.VERTEX_BINDING_DESCRIPTION[0],
    vertexAttributeDescriptionCount = 1,
    pVertexAttributeDescriptions    = raw_data(geometry.VERTEX_ATTRIBUTE_DESCRIPTIONS[:]),
  }

  // Create point light pipeline
  {
    vert_module := gpu.create_shader_module(gctx.device, SHADER_POINT_VERT) or_return
    defer vk.DestroyShaderModule(gctx.device, vert_module, nil)
    frag_module := gpu.create_shader_module(gctx.device, SHADER_POINT_FRAG) or_return
    defer vk.DestroyShaderModule(gctx.device, frag_module, nil)
    shader_stages := gpu.create_vert_frag_stages(vert_module, frag_module, &shared.SHADER_SPEC_CONSTANTS)
    pipeline_info := vk.GraphicsPipelineCreateInfo {
      sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
      pNext               = &gpu.HDR_STANDARD_RENDERING_INFO,
      stageCount          = len(shader_stages),
      pStages             = raw_data(shader_stages[:]),
      pVertexInputState   = &vertex_input,
      pInputAssemblyState = &gpu.STANDARD_INPUT_ASSEMBLY,
      pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
      pRasterizationState = &gpu.STANDARD_RASTERIZER,
      pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
      pColorBlendState    = &gpu.COLOR_BLENDING_OVERFLOW,
      pDynamicState       = &dynamic_state,
      pDepthStencilState  = &gpu.READ_ONLY_DEPTH_STATE,
      layout              = self.pipeline_layout,
    }
    vk.CreateGraphicsPipelines(gctx.device, 0, 1, &pipeline_info, nil, &self.point_pipeline) or_return
    defer if ret != .SUCCESS do vk.DestroyPipeline(gctx.device, self.point_pipeline, nil)
  }

  // Create spot light pipeline
  {
    vert_module := gpu.create_shader_module(gctx.device, SHADER_SPOT_VERT) or_return
    defer vk.DestroyShaderModule(gctx.device, vert_module, nil)
    frag_module := gpu.create_shader_module(gctx.device, SHADER_SPOT_FRAG) or_return
    defer vk.DestroyShaderModule(gctx.device, frag_module, nil)
    shader_stages := gpu.create_vert_frag_stages(vert_module, frag_module, &shared.SHADER_SPEC_CONSTANTS)
    pipeline_info := vk.GraphicsPipelineCreateInfo {
      sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
      pNext               = &gpu.HDR_STANDARD_RENDERING_INFO,
      stageCount          = len(shader_stages),
      pStages             = raw_data(shader_stages[:]),
      pVertexInputState   = &vertex_input,
      pInputAssemblyState = &gpu.STANDARD_INPUT_ASSEMBLY,
      pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
      pRasterizationState = &gpu.STANDARD_RASTERIZER,
      pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
      pColorBlendState    = &gpu.COLOR_BLENDING_OVERFLOW,
      pDynamicState       = &dynamic_state,
      pDepthStencilState  = &gpu.READ_ONLY_DEPTH_STATE,
      layout              = self.pipeline_layout,
    }
    vk.CreateGraphicsPipelines(gctx.device, 0, 1, &pipeline_info, nil, &self.spot_pipeline) or_return
    defer if ret != .SUCCESS do vk.DestroyPipeline(gctx.device, self.spot_pipeline, nil)
  }

  // Create directional light pipeline
  {
    vert_module := gpu.create_shader_module(gctx.device, SHADER_DIRECTIONAL_VERT) or_return
    defer vk.DestroyShaderModule(gctx.device, vert_module, nil)
    frag_module := gpu.create_shader_module(gctx.device, SHADER_DIRECTIONAL_FRAG) or_return
    defer vk.DestroyShaderModule(gctx.device, frag_module, nil)
    shader_stages := gpu.create_vert_frag_stages(vert_module, frag_module, &shared.SHADER_SPEC_CONSTANTS)
    pipeline_info := vk.GraphicsPipelineCreateInfo {
      sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
      pNext               = &gpu.HDR_STANDARD_RENDERING_INFO,
      stageCount          = len(shader_stages),
      pStages             = raw_data(shader_stages[:]),
      pVertexInputState   = &vertex_input,
      pInputAssemblyState = &gpu.STANDARD_INPUT_ASSEMBLY,
      pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
      pRasterizationState = &gpu.STANDARD_RASTERIZER,
      pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
      pColorBlendState    = &gpu.COLOR_BLENDING_OVERFLOW,
      pDynamicState       = &dynamic_state,
      pDepthStencilState  = &gpu.READ_ONLY_DEPTH_STATE,
      layout              = self.pipeline_layout,
    }
    vk.CreateGraphicsPipelines(gctx.device, 0, 1, &pipeline_info, nil, &self.directional_pipeline) or_return
    defer if ret != .SUCCESS do vk.DestroyPipeline(gctx.device, self.directional_pipeline, nil)
  }

  log.info("Direct lighting pipelines initialized successfully (point, spot, directional)")
  return .SUCCESS
}

setup :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) -> (ret: vk.Result) {
  create_light_volume_mesh(
    gctx,
    &self.sphere_mesh,
    geometry.make_sphere(),
  ) or_return
  defer if ret != .SUCCESS {
    destroy_light_volume_mesh(gctx.device, &self.sphere_mesh)
  }
  create_light_volume_mesh(
    gctx,
    &self.cone_mesh,
    geometry.make_cone(),
  ) or_return
  defer if ret != .SUCCESS {
    destroy_light_volume_mesh(gctx.device, &self.cone_mesh)
  }
  create_light_volume_mesh(
    gctx,
    &self.triangle_mesh,
    geometry.make_fullscreen_triangle(),
  ) or_return
  defer if ret != .SUCCESS {
    destroy_light_volume_mesh(gctx.device, &self.triangle_mesh)
  }
  log.info("Direct lighting GPU resources setup")
  return .SUCCESS
}

teardown :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) {
  destroy_light_volume_mesh(gctx.device, &self.sphere_mesh)
  destroy_light_volume_mesh(gctx.device, &self.cone_mesh)
  destroy_light_volume_mesh(gctx.device, &self.triangle_mesh)
}

shutdown :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) {
  vk.DestroyPipelineLayout(gctx.device, self.pipeline_layout, nil)
  vk.DestroyPipeline(gctx.device, self.point_pipeline, nil)
  vk.DestroyPipeline(gctx.device, self.spot_pipeline, nil)
  vk.DestroyPipeline(gctx.device, self.directional_pipeline, nil)
}

begin_pass :: proc(
  self: ^Renderer,
  final_image_handle: gpu.Texture2DHandle,
  depth_handle: gpu.Texture2DHandle,
  texture_manager: ^gpu.TextureManager,
  command_buffer: vk.CommandBuffer,
  cameras_descriptor_set: vk.DescriptorSet,
) {
  final_image := gpu.get_texture_2d(
    texture_manager,
    final_image_handle,
  )
  depth_texture := gpu.get_texture_2d(
    texture_manager,
    depth_handle,
  )
  gpu.begin_rendering(
    command_buffer,
    depth_texture.spec.extent,
    gpu.create_depth_attachment(
      depth_texture,
      .LOAD,
      .DONT_CARE,
    ),
    gpu.create_color_attachment(final_image, .LOAD, .STORE, BG_BLUE_GRAY),
  )
  gpu.set_viewport_scissor(command_buffer, depth_texture.spec.extent)
  // Bind descriptor sets (shared across all light types)
  // Pipeline binding happens per-light in render_*_light() functions
  descriptor_sets := [?]vk.DescriptorSet{cameras_descriptor_set, texture_manager.descriptor_set}
  vk.CmdBindDescriptorSets(
    command_buffer,
    .GRAPHICS,
    self.pipeline_layout,
    0,
    2,
    raw_data(descriptor_sets[:]),
    0,
    nil,
  )
}

@(private)
draw_light_volume_mesh :: proc(
  mesh: ^LightVolumeMesh,
  command_buffer: vk.CommandBuffer,
) {
  gpu.bind_vertex_index_buffers(
    command_buffer,
    mesh.vertex_buffer.buffer,
    mesh.index_buffer.buffer,
    0,
    0,
  )
  vk.CmdDrawIndexed(command_buffer, mesh.index_count, 1, 0, 0, 0)
}

@(private)
push_and_draw :: proc(
  self: ^Renderer,
  command_buffer: vk.CommandBuffer,
  pipeline: vk.Pipeline,
  push: rawptr,
  push_size: int,
  mesh: ^LightVolumeMesh,
  depth_compare_op: vk.CompareOp,
  cull_mode: vk.CullModeFlags,
) {
  vk.CmdBindPipeline(command_buffer, .GRAPHICS, pipeline)
  vk.CmdPushConstants(
    command_buffer,
    self.pipeline_layout,
    {.VERTEX, .FRAGMENT},
    0,
    u32(push_size),
    push,
  )
  vk.CmdSetDepthCompareOp(command_buffer, depth_compare_op)
  vk.CmdSetCullMode(command_buffer, cull_mode)
  draw_light_volume_mesh(mesh, command_buffer)
}

render_point_light :: proc(
  self: ^Renderer,
  camera_handle: u32,
  position_texture_idx: u32,
  normal_texture_idx: u32,
  albedo_texture_idx: u32,
  metallic_texture_idx: u32,
  light_color: [4]f32,
  position: [3]f32,
  radius: f32,
  shadow_map_idx: u32,
  shadow_view_projection: matrix[4, 4]f32,
  command_buffer: vk.CommandBuffer,
) {
  push := PointLightPushConstants{
    scene_camera_idx       = camera_handle,
    position_texture_index = position_texture_idx,
    normal_texture_index   = normal_texture_idx,
    albedo_texture_index   = albedo_texture_idx,
    metallic_texture_index = metallic_texture_idx,
    shadow_map_idx         = shadow_map_idx,
    light_color            = light_color,
    position               = position,
    radius                 = radius,
  }
  if shadow_map_idx != 0xFFFFFFFF {
    push.shadow_view_projection = shadow_view_projection
  }
  push_and_draw(
    self,
    command_buffer,
    self.point_pipeline,
    &push,
    size_of(push),
    &self.sphere_mesh,
    .GREATER_OR_EQUAL,
    {.FRONT},
  )
}

render_spot_light :: proc(
  self: ^Renderer,
  camera_handle: u32,
  position_texture_idx: u32,
  normal_texture_idx: u32,
  albedo_texture_idx: u32,
  metallic_texture_idx: u32,
  light_color: [4]f32,
  position: [3]f32,
  direction: [3]f32,
  radius: f32,
  angle_inner: f32,
  angle_outer: f32,
  shadow_map_idx: u32,
  shadow_view_projection: matrix[4, 4]f32,
  command_buffer: vk.CommandBuffer,
) {
  // Pack indices: 2 indices per u32 (16-bit each)
  shadow_and_camera := (shadow_map_idx & 0xFFFF) | ((camera_handle & 0xFFFF) << 16)
  position_and_normal := (position_texture_idx & 0xFFFF) | ((normal_texture_idx & 0xFFFF) << 16)
  albedo_and_metallic := (albedo_texture_idx & 0xFFFF) | ((metallic_texture_idx & 0xFFFF) << 16)

  push := SpotLightPushConstants{
    light_color                  = light_color,
    position                     = position,
    direction                    = linalg.normalize(direction),
    radius                       = radius,
    angle_inner                  = angle_inner,
    angle_outer                  = angle_outer,
    shadow_and_camera_indices    = shadow_and_camera,
    position_and_normal_indices  = position_and_normal,
    albedo_and_metallic_indices  = albedo_and_metallic,
  }
  if shadow_map_idx != 0xFFFFFFFF {
    push.shadow_view_projection = shadow_view_projection
  }
  push_and_draw(
    self,
    command_buffer,
    self.spot_pipeline,
    &push,
    size_of(push),
    &self.cone_mesh,
    .GREATER_OR_EQUAL,
    {.BACK},
  )
}

render_directional_light :: proc(
  self: ^Renderer,
  camera_handle: u32,
  position_texture_idx: u32,
  normal_texture_idx: u32,
  albedo_texture_idx: u32,
  metallic_texture_idx: u32,
  light_color: [4]f32,
  direction: [3]f32,
  shadow_map_idx: u32,
  shadow_view_projection: matrix[4, 4]f32,
  command_buffer: vk.CommandBuffer,
) {
  push := DirectionalLightPushConstants{
    scene_camera_idx       = camera_handle,
    position_texture_index = position_texture_idx,
    normal_texture_index   = normal_texture_idx,
    albedo_texture_index   = albedo_texture_idx,
    metallic_texture_index = metallic_texture_idx,
    shadow_map_idx         = shadow_map_idx,
    light_color            = light_color,
    direction              = direction,
  }
  if shadow_map_idx != 0xFFFFFFFF {
    push.shadow_view_projection = shadow_view_projection
  }
  push_and_draw(
    self,
    command_buffer,
    self.directional_pipeline,
    &push,
    size_of(push),
    &self.triangle_mesh,
    .ALWAYS,
    {.BACK},
  )
}

end_pass :: proc(command_buffer: vk.CommandBuffer) {
  vk.CmdEndRendering(command_buffer)
}
