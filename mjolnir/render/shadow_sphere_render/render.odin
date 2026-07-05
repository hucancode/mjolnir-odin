package shadow_sphere_render

import "../../geometry"
import "../../gpu"
import vk "vendor:vulkan"

SHADER_SPHERE_DEPTH_VERT :: #load("../../shader/shadow_spherical/vert.spv")
SHADER_SPHERE_DEPTH_GEOM :: #load("../../shader/shadow_spherical/geom.spv")
SHADER_SPHERE_DEPTH_FRAG :: #load("../../shader/shadow_spherical/frag.spv")

ShadowTransform :: struct {
  view:            matrix[4, 4]f32,
  projection:      matrix[4, 4]f32,
  view_projection: matrix[4, 4]f32,
  near:            f32,
  far:             f32,
  frustum_planes:  [6][4]f32,
  position:        [3]f32,  // Light position for cubemap generation
}

ShadowDepthPushConstants :: struct {
  projection:     matrix[4, 4]f32,  // 64 bytes (aligned to 16 bytes)
  light_position: [3]f32,           // 12 bytes
  near_plane:     f32,              // 4 bytes
  far_plane:      f32,              // 4 bytes
}
// Total: 84 bytes (std140 layout)

System :: struct {
  max_draws:             u32,
  shadow_map_size:       u32,
  depth_pipeline_layout: vk.PipelineLayout,
  depth_pipeline:        vk.Pipeline,
}

init :: proc(
  self: ^System,
  gctx: ^gpu.GPUContext,
  textures_set_layout: vk.DescriptorSetLayout,
  bone_set_layout: vk.DescriptorSetLayout,
  material_set_layout: vk.DescriptorSetLayout,
  node_data_set_layout: vk.DescriptorSetLayout,
  mesh_data_set_layout: vk.DescriptorSetLayout,
  vertex_skinning_set_layout: vk.DescriptorSetLayout,
  max_draws: u32,
  shadow_map_size: u32,
) -> (
  ret: vk.Result,
) {
  self.max_draws = max_draws
  self.shadow_map_size = shadow_map_size
  self.depth_pipeline_layout = gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange {
      stageFlags = {.VERTEX, .GEOMETRY, .FRAGMENT},
      size = size_of(ShadowDepthPushConstants),
    },
    textures_set_layout,
    bone_set_layout,
    material_set_layout,
    node_data_set_layout,
    mesh_data_set_layout,
    vertex_skinning_set_layout,
  ) or_return
  defer if ret != .SUCCESS {
    vk.DestroyPipelineLayout(gctx.device, self.depth_pipeline_layout, nil)
    self.depth_pipeline_layout = 0
  }
  vert := gpu.create_shader_module(
    gctx.device,
    SHADER_SPHERE_DEPTH_VERT,
  ) or_return
  defer vk.DestroyShaderModule(gctx.device, vert, nil)
  geom := gpu.create_shader_module(
    gctx.device,
    SHADER_SPHERE_DEPTH_GEOM,
  ) or_return
  defer vk.DestroyShaderModule(gctx.device, geom, nil)
  frag := gpu.create_shader_module(
    gctx.device,
    SHADER_SPHERE_DEPTH_FRAG,
  ) or_return
  defer vk.DestroyShaderModule(gctx.device, frag, nil)

  vertex_bindings := [?]vk.VertexInputBindingDescription {
    {binding = 0, stride = size_of(geometry.Vertex), inputRate = .VERTEX},
  }
  vertex_attributes := [?]vk.VertexInputAttributeDescription {
    {
      location = 0,
      binding = 0,
      format = .R32G32B32_SFLOAT,
      offset = u32(offset_of(geometry.Vertex, position)),
    },
  }
  vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
    sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    vertexBindingDescriptionCount   = len(vertex_bindings),
    pVertexBindingDescriptions      = raw_data(vertex_bindings[:]),
    vertexAttributeDescriptionCount = len(vertex_attributes),
    pVertexAttributeDescriptions    = raw_data(vertex_attributes[:]),
  }
  stages := gpu.create_vert_geo_frag_stages(vert, geom, frag)
  info := vk.GraphicsPipelineCreateInfo {
    sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
    pNext               = &gpu.DEPTH_ONLY_RENDERING_INFO,
    stageCount          = len(stages),
    pStages             = raw_data(stages[:]),
    pVertexInputState   = &vertex_input_info,
    pInputAssemblyState = &gpu.STANDARD_INPUT_ASSEMBLY,
    pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
    pRasterizationState = &gpu.INVERSE_RASTERIZER,
    pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
    pDepthStencilState  = &gpu.READ_WRITE_DEPTH_STATE,
    pDynamicState       = &gpu.STANDARD_DYNAMIC_STATES,
    layout              = self.depth_pipeline_layout,
  }
  vk.CreateGraphicsPipelines(
    gctx.device,
    0,
    1,
    &info,
    nil,
    &self.depth_pipeline,
  ) or_return
  defer if ret != .SUCCESS {
    vk.DestroyPipeline(gctx.device, self.depth_pipeline, nil)
    self.depth_pipeline = 0
  }
  return .SUCCESS
}

shutdown :: proc(self: ^System, gctx: ^gpu.GPUContext) {
  vk.DestroyPipeline(gctx.device, self.depth_pipeline, nil)
  vk.DestroyPipelineLayout(gctx.device, self.depth_pipeline_layout, nil)
}

render :: proc(
  self: ^System,
  command_buffer: vk.CommandBuffer,
  texture_manager: ^gpu.TextureManager,
  projection: matrix[4,4]f32,
  near, far: f32,
  position: [3]f32,
  shadow_map: gpu.TextureCubeHandle,
  draw_command: gpu.MutableBuffer(vk.DrawIndexedIndirectCommand),
  draw_count: gpu.MutableBuffer(u32),
  textures_descriptor_set: vk.DescriptorSet,
  bone_descriptor_set: vk.DescriptorSet,
  material_descriptor_set: vk.DescriptorSet,
  node_data_descriptor_set: vk.DescriptorSet,
  mesh_data_descriptor_set: vk.DescriptorSet,
  vertex_skinning_descriptor_set: vk.DescriptorSet,
  vertex_buffer: vk.Buffer,
  index_buffer: vk.Buffer,
  frame_index: u32,
) {
  gpu.buffer_barrier(
    command_buffer,
    draw_command.buffer,
    vk.DeviceSize(draw_command.bytes_count),
    {.SHADER_WRITE},
    {.INDIRECT_COMMAND_READ},
    {.COMPUTE_SHADER},
    {.DRAW_INDIRECT},
  )
  gpu.buffer_barrier(
    command_buffer,
    draw_count.buffer,
    vk.DeviceSize(draw_count.bytes_count),
    {.SHADER_WRITE},
    {.INDIRECT_COMMAND_READ},
    {.COMPUTE_SHADER},
    {.DRAW_INDIRECT},
  )
  depth_cube := gpu.get_texture_cube(texture_manager, shadow_map)
  if depth_cube == nil do return
  gpu.image_discard_barrier(
    command_buffer,
    depth_cube.image,
    {.DEPTH_STENCIL_ATTACHMENT_WRITE},
    {.EARLY_FRAGMENT_TESTS},
    {.DEPTH},
    layer_count = 6,
  )
  depth_attachment := gpu.create_cube_depth_attachment(
    depth_cube,
    .CLEAR,
    .STORE,
  )
  gpu.begin_depth_rendering(
    command_buffer,
    vk.Extent2D{self.shadow_map_size, self.shadow_map_size},
    &depth_attachment,
    layer_count = 6,
  )
  gpu.set_viewport_scissor(
    command_buffer,
    vk.Extent2D{self.shadow_map_size, self.shadow_map_size},
    flip_y = false,
  )
  gpu.bind_graphics_pipeline(
    command_buffer,
    self.depth_pipeline,
    self.depth_pipeline_layout,
    textures_descriptor_set,
    bone_descriptor_set,
    material_descriptor_set,
    node_data_descriptor_set,
    mesh_data_descriptor_set,
    vertex_skinning_descriptor_set,
  )
  push := ShadowDepthPushConstants{
    projection     = projection,
    near_plane     = near,
    far_plane      = far,
    light_position = position,
  }
  // Flip clip-space X to match Vulkan cube-map sampling convention. Vulkan
  // disallows negative viewport width, so the flip lives in the projection
  // (paired with INVERSE_RASTERIZER above to keep winding correct).
  push.projection[0, 0] = -push.projection[0, 0]
  push.projection[0, 1] = -push.projection[0, 1]
  push.projection[0, 2] = -push.projection[0, 2]
  push.projection[0, 3] = -push.projection[0, 3]
  vk.CmdPushConstants(
    command_buffer,
    self.depth_pipeline_layout,
    {.VERTEX, .GEOMETRY, .FRAGMENT},
    0,
    size_of(push),
    &push,
  )
  gpu.bind_vertex_index_buffers(command_buffer, vertex_buffer, index_buffer)
  vk.CmdDrawIndexedIndirectCount(
    command_buffer,
    draw_command.buffer,
    0,
    draw_count.buffer,
    0,
    self.max_draws,
    u32(size_of(vk.DrawIndexedIndirectCommand)),
  )
  vk.CmdEndRendering(command_buffer)
  gpu.memory_barrier(
    command_buffer,
    {.DEPTH_STENCIL_ATTACHMENT_WRITE},
    {.SHADER_READ},
    {.LATE_FRAGMENT_TESTS},
    {.FRAGMENT_SHADER},
  )
}
