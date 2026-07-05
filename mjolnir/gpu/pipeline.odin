package gpu

import "core:slice"
import vk "vendor:vulkan"

VERTEX_INPUT_NONE := vk.PipelineVertexInputStateCreateInfo {
  sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
}

READ_WRITE_DEPTH_STATE := vk.PipelineDepthStencilStateCreateInfo {
  sType            = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  depthTestEnable  = true,
  depthWriteEnable = true,
  depthCompareOp   = .LESS_OR_EQUAL,
}

READ_ONLY_DEPTH_STATE := vk.PipelineDepthStencilStateCreateInfo {
  sType           = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  depthTestEnable = true,
  depthCompareOp  = .LESS_OR_EQUAL,
}

READ_ONLY_INVERSE_DEPTH_STATE := vk.PipelineDepthStencilStateCreateInfo {
  sType           = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  depthTestEnable = true,
  depthCompareOp  = .GREATER_OR_EQUAL,
}

DISABLED_DEPTH_STATE := vk.PipelineDepthStencilStateCreateInfo {
  sType            = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  depthTestEnable  = false,
  depthWriteEnable = false,
}

DYNAMIC_STATES := [?]vk.DynamicState{.VIEWPORT, .SCISSOR}

STANDARD_DYNAMIC_STATES := vk.PipelineDynamicStateCreateInfo {
  sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
  dynamicStateCount = len(DYNAMIC_STATES),
  pDynamicStates    = raw_data(DYNAMIC_STATES[:]),
}

STANDARD_RASTERIZER := vk.PipelineRasterizationStateCreateInfo {
  sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  polygonMode = .FILL,
  cullMode    = {.BACK},
  frontFace   = .COUNTER_CLOCKWISE,
}

INVERSE_RASTERIZER := vk.PipelineRasterizationStateCreateInfo {
  sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  polygonMode = .FILL,
  cullMode    = {.BACK},
  frontFace   = .CLOCKWISE,
}

DOUBLE_SIDED_RASTERIZER := vk.PipelineRasterizationStateCreateInfo {
  sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  polygonMode = .FILL,
  lineWidth   = 1.0,
}

LINE_RASTERIZER := vk.PipelineRasterizationStateCreateInfo {
  sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  polygonMode = .LINE,
  lineWidth   = 1.0,
}

BOLD_DOUBLE_SIDED_RASTERIZER := vk.PipelineRasterizationStateCreateInfo {
  sType       = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
  polygonMode = .FILL,
  lineWidth   = 3.0,
}

STANDARD_INPUT_ASSEMBLY := vk.PipelineInputAssemblyStateCreateInfo {
  sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
  topology = .TRIANGLE_LIST,
}

POINT_INPUT_ASSEMBLY := vk.PipelineInputAssemblyStateCreateInfo {
  sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
  topology = .POINT_LIST,
}

LINE_INPUT_ASSEMBLY := vk.PipelineInputAssemblyStateCreateInfo {
  sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
  topology = .LINE_STRIP,
}

STANDARD_VIEWPORT_STATE := vk.PipelineViewportStateCreateInfo {
  sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
  viewportCount = 1,
  scissorCount  = 1,
}

STANDARD_MULTISAMPLING := vk.PipelineMultisampleStateCreateInfo {
  sType                = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
  rasterizationSamples = {._1},
}

BLEND_OVERRIDE := vk.PipelineColorBlendAttachmentState {
  colorWriteMask = {.R, .G, .B, .A},
}

COLOR_BLENDING_OVERRIDE := vk.PipelineColorBlendStateCreateInfo {
  sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
  attachmentCount = 1,
  pAttachments    = &BLEND_OVERRIDE,
}

BLEND_ADDITIVE := vk.PipelineColorBlendAttachmentState {
  blendEnable         = true,
  srcColorBlendFactor = .SRC_ALPHA,
  dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
  colorBlendOp        = .ADD,
  srcAlphaBlendFactor = .SRC_ALPHA,
  dstAlphaBlendFactor = .ONE_MINUS_SRC_ALPHA,
  alphaBlendOp        = .ADD,
  colorWriteMask      = {.R, .G, .B, .A},
}

COLOR_BLENDING_ADDITIVE := vk.PipelineColorBlendStateCreateInfo {
  sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
  attachmentCount = 1,
  pAttachments    = &BLEND_ADDITIVE,
}

BLEND_OVERFLOW := vk.PipelineColorBlendAttachmentState {
  blendEnable         = true,
  srcColorBlendFactor = .ONE,
  dstColorBlendFactor = .ONE,
  colorBlendOp        = .ADD,
  srcAlphaBlendFactor = .ONE,
  dstAlphaBlendFactor = .ONE,
  alphaBlendOp        = .ADD,
  colorWriteMask      = {.R, .G, .B, .A},
}

COLOR_BLENDING_OVERFLOW := vk.PipelineColorBlendStateCreateInfo {
  sType           = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
  attachmentCount = 1,
  pAttachments    = &BLEND_OVERFLOW,
}

STANDARD_COLOR_FORMAT := vk.Format.B8G8R8A8_SRGB
STANDARD_RENDERING_INFO := vk.PipelineRenderingCreateInfo {
  sType                   = .PIPELINE_RENDERING_CREATE_INFO,
  colorAttachmentCount    = 1,
  pColorAttachmentFormats = &STANDARD_COLOR_FORMAT,
  depthAttachmentFormat   = .D32_SFLOAT,
}

COLOR_ONLY_RENDERING_INFO := vk.PipelineRenderingCreateInfo {
  sType                   = .PIPELINE_RENDERING_CREATE_INFO,
  colorAttachmentCount    = 1,
  pColorAttachmentFormats = &STANDARD_COLOR_FORMAT,
}

HDR_COLOR_FORMAT := vk.Format.R16G16B16A16_SFLOAT
HDR_STANDARD_RENDERING_INFO := vk.PipelineRenderingCreateInfo {
  sType                   = .PIPELINE_RENDERING_CREATE_INFO,
  colorAttachmentCount    = 1,
  pColorAttachmentFormats = &HDR_COLOR_FORMAT,
  depthAttachmentFormat   = .D32_SFLOAT,
}

HDR_COLOR_ONLY_RENDERING_INFO := vk.PipelineRenderingCreateInfo {
  sType                   = .PIPELINE_RENDERING_CREATE_INFO,
  colorAttachmentCount    = 1,
  pColorAttachmentFormats = &HDR_COLOR_FORMAT,
}

DEPTH_ONLY_RENDERING_INFO := vk.PipelineRenderingCreateInfo {
  sType                 = .PIPELINE_RENDERING_CREATE_INFO,
  depthAttachmentFormat = .D32_SFLOAT,
}

create_vert_frag_stages :: proc(
  vert_module: vk.ShaderModule,
  frag_module: vk.ShaderModule,
  specialization: ^vk.SpecializationInfo = nil,
) -> [2]vk.PipelineShaderStageCreateInfo {
  return [2]vk.PipelineShaderStageCreateInfo {
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.VERTEX},
      module = vert_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.FRAGMENT},
      module = frag_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
  }
}

create_vert_stage :: proc(
  vert_module: vk.ShaderModule,
  specialization: ^vk.SpecializationInfo = nil,
) -> [1]vk.PipelineShaderStageCreateInfo {
  return [1]vk.PipelineShaderStageCreateInfo {
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.VERTEX},
      module = vert_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
  }
}

create_frag_stage :: proc(
  frag_module: vk.ShaderModule,
  specialization: ^vk.SpecializationInfo = nil,
) -> [1]vk.PipelineShaderStageCreateInfo {
  return [1]vk.PipelineShaderStageCreateInfo {
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.FRAGMENT},
      module = frag_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
  }
}

create_vert_geo_frag_stages :: proc(
  vert_module: vk.ShaderModule,
  geo_module: vk.ShaderModule,
  frag_module: vk.ShaderModule,
  specialization: ^vk.SpecializationInfo = nil,
) -> [3]vk.PipelineShaderStageCreateInfo {
  return [3]vk.PipelineShaderStageCreateInfo {
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.VERTEX},
      module = vert_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.GEOMETRY},
      module = geo_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
    {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.FRAGMENT},
      module = frag_module,
      pName = "main",
      pSpecializationInfo = specialization,
    },
  }
}

begin_record :: proc(
  command_buffer: vk.CommandBuffer,
  flags: vk.CommandBufferUsageFlags = {.ONE_TIME_SUBMIT},
) -> vk.Result {
  vk.ResetCommandBuffer(command_buffer, {}) or_return
  return vk.BeginCommandBuffer(
    command_buffer,
    &{sType = .COMMAND_BUFFER_BEGIN_INFO, flags = flags},
  )
}

end_record :: proc(command_buffer: vk.CommandBuffer) -> vk.Result {
  return vk.EndCommandBuffer(command_buffer)
}

create_dynamic_state :: proc(
  states: []vk.DynamicState,
) -> vk.PipelineDynamicStateCreateInfo {
  return {
    sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    dynamicStateCount = u32(len(states)),
    pDynamicStates = raw_data(states),
  }
}

allocate_descriptor_set_single :: proc(
  gctx: ^GPUContext,
  ret: ^vk.DescriptorSet,
  layout: ^vk.DescriptorSetLayout,
) -> vk.Result {
  return vk.AllocateDescriptorSets(
    gctx.device,
    &{
      sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
      descriptorPool = gctx.descriptor_pool,
      descriptorSetCount = 1,
      pSetLayouts = layout,
    },
    ret,
  )
}

allocate_descriptor_set_multi :: proc(
  gctx: ^GPUContext,
  ret: []vk.DescriptorSet,
  layout: vk.DescriptorSetLayout,
) -> vk.Result {
  layouts := make([]vk.DescriptorSetLayout, len(ret))
  defer delete(layouts)
  slice.fill(layouts, layout)
  return vk.AllocateDescriptorSets(
    gctx.device,
    &{
      sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
      descriptorPool = gctx.descriptor_pool,
      descriptorSetCount = u32(len(layouts)),
      pSetLayouts = raw_data(layouts),
    },
    raw_data(ret),
  )
}

allocate_descriptor_set :: proc {
  allocate_descriptor_set_single,
  allocate_descriptor_set_multi,
}

allocate_command_buffer_single :: proc(
  gctx: ^GPUContext,
  level: vk.CommandBufferLevel = .PRIMARY,
) -> (
  command: vk.CommandBuffer,
  ret: vk.Result,
) {
  vk.AllocateCommandBuffers(
    gctx.device,
    &vk.CommandBufferAllocateInfo {
      sType = .COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool = gctx.command_pool,
      level = level,
      commandBufferCount = 1,
    },
    &command,
  ) or_return
  return
}

allocate_command_buffer_multi :: proc(
  gctx: ^GPUContext,
  out_commands: []vk.CommandBuffer,
  level: vk.CommandBufferLevel = .PRIMARY,
) -> vk.Result {
  return vk.AllocateCommandBuffers(
    gctx.device,
    &vk.CommandBufferAllocateInfo {
      sType = .COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool = gctx.command_pool,
      level = level,
      commandBufferCount = u32(len(out_commands)),
    },
    raw_data(out_commands),
  )
}

allocate_command_buffer :: proc {
  allocate_command_buffer_single,
  allocate_command_buffer_multi,
}

allocate_compute_command_buffer_single :: proc(
  gctx: ^GPUContext,
  level: vk.CommandBufferLevel = .PRIMARY,
) -> (
  command: vk.CommandBuffer,
  ret: vk.Result,
) {
  pool, ok := gctx.compute_command_pool.?
  if !ok do return {}, .ERROR_UNKNOWN
  vk.AllocateCommandBuffers(
    gctx.device,
    &{
      sType = .COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool = pool,
      level = level,
      commandBufferCount = 1,
    },
    &command,
  ) or_return
  return
}

allocate_compute_command_buffer_multi :: proc(
  gctx: ^GPUContext,
  out_commands: []vk.CommandBuffer,
  level: vk.CommandBufferLevel = .PRIMARY,
) -> vk.Result {
  pool, ok := gctx.compute_command_pool.?
  if !ok do return .ERROR_UNKNOWN
  vk.AllocateCommandBuffers(
    gctx.device,
    &{
      sType = .COMMAND_BUFFER_ALLOCATE_INFO,
      commandPool = pool,
      level = level,
      commandBufferCount = u32(len(out_commands)),
    },
    raw_data(out_commands),
  ) or_return
  return .SUCCESS
}

allocate_compute_command_buffer :: proc {
  allocate_compute_command_buffer_single,
  allocate_compute_command_buffer_multi,
}

free_command_buffer :: proc(gctx: ^GPUContext, commands: ..vk.CommandBuffer) {
  vk.FreeCommandBuffers(
    gctx.device,
    gctx.command_pool,
    u32(len(commands)),
    raw_data(commands),
  )
}

free_compute_command_buffer_single :: proc(
  gctx: ^GPUContext,
  command: ^vk.CommandBuffer,
) {
  pool, ok := gctx.compute_command_pool.?
  if !ok do return
  vk.FreeCommandBuffers(gctx.device, pool, 1, command)
}

free_compute_command_buffer_multi :: proc(
  gctx: ^GPUContext,
  commands: []vk.CommandBuffer,
) {
  pool, ok := gctx.compute_command_pool.?
  if !ok do return
  vk.FreeCommandBuffers(
    gctx.device,
    pool,
    u32(len(commands)),
    raw_data(commands),
  )
}

free_compute_command_buffer :: proc {
  free_compute_command_buffer_single,
  free_compute_command_buffer_multi,
}

update_descriptor_set :: proc(
  gctx: ^GPUContext,
  dst: vk.DescriptorSet,
  buffers: ..struct {
    type: vk.DescriptorType,
    info: union {
      vk.DescriptorBufferInfo,
      vk.DescriptorImageInfo,
    },
  },
) {
  writes := make([]vk.WriteDescriptorSet, len(buffers))
  defer delete(writes)
  for &b, i in buffers {
    writes[i] = {
      sType           = .WRITE_DESCRIPTOR_SET,
      dstSet          = dst,
      dstBinding      = u32(i),
      descriptorType  = b.type,
      descriptorCount = 1,
    }
    switch &v in b.info {
    case vk.DescriptorBufferInfo:
      writes[i].pBufferInfo = &v
    case vk.DescriptorImageInfo:
      writes[i].pImageInfo = &v
    }
  }
  vk.UpdateDescriptorSets(
    gctx.device,
    u32(len(writes)),
    raw_data(writes),
    0,
    nil,
  )
}

update_descriptor_set_array :: proc(
  gctx: ^GPUContext,
  dst: vk.DescriptorSet,
  dst_binding: u32,
  buffers: ..struct {
    type: vk.DescriptorType,
    info: union {
      vk.DescriptorBufferInfo,
      vk.DescriptorImageInfo,
    },
  },
) {
  update_descriptor_set_array_offset(gctx, dst, dst_binding, 0, ..buffers)
}

update_descriptor_set_array_offset :: proc(
  gctx: ^GPUContext,
  dst: vk.DescriptorSet,
  dst_binding: u32,
  dst_offset: u32,
  buffers: ..struct {
    type: vk.DescriptorType,
    info: union {
      vk.DescriptorBufferInfo,
      vk.DescriptorImageInfo,
    },
  },
) {
  writes := make([]vk.WriteDescriptorSet, len(buffers))
  defer delete(writes)
  for &b, i in buffers {
    writes[i] = {
      sType           = .WRITE_DESCRIPTOR_SET,
      dstSet          = dst,
      dstBinding      = dst_binding,
      dstArrayElement = u32(i) + dst_offset,
      descriptorType  = b.type,
      descriptorCount = 1,
    }
    switch &v in b.info {
    case vk.DescriptorBufferInfo:
      writes[i].pBufferInfo = &v
    case vk.DescriptorImageInfo:
      writes[i].pImageInfo = &v
    }
  }
  vk.UpdateDescriptorSets(
    gctx.device,
    u32(len(writes)),
    raw_data(writes),
    0,
    nil,
  )
}

create_descriptor_set :: proc(
  gctx: ^GPUContext,
  layout: ^vk.DescriptorSetLayout,
  buffers: ..struct {
    type: vk.DescriptorType,
    info: union {
      vk.DescriptorBufferInfo,
      vk.DescriptorImageInfo,
    },
  },
) -> (
  dst: vk.DescriptorSet,
  ret: vk.Result,
) {
  allocate_descriptor_set_single(gctx, &dst, layout) or_return
  update_descriptor_set(gctx, dst, ..buffers)
  return dst, .SUCCESS
}

create_descriptor_set_layout :: proc(gctx: ^GPUContext, bindings: ..struct {
    type:  vk.DescriptorType,
    flags: vk.ShaderStageFlags,
  }) -> (layout: vk.DescriptorSetLayout, ret: vk.Result) {
  vk_bindings := make([]vk.DescriptorSetLayoutBinding, len(bindings))
  defer delete(vk_bindings)
  for &b, i in bindings {
    vk_bindings[i] = {
      binding         = u32(i),
      descriptorType  = b.type,
      descriptorCount = 1,
      stageFlags      = b.flags,
    }
  }
  vk.CreateDescriptorSetLayout(
    gctx.device,
    &{
      sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      bindingCount = u32(len(vk_bindings)),
      pBindings = raw_data(vk_bindings),
    },
    nil,
    &layout,
  ) or_return
  ret = .SUCCESS
  return
}

create_descriptor_set_layout_array :: proc(
  gctx: ^GPUContext,
  bindings: ..struct {
    type:  vk.DescriptorType,
    count: u32,
    flags: vk.ShaderStageFlags,
  },
) -> (
  layout: vk.DescriptorSetLayout,
  ret: vk.Result,
) {
  vk_bindings := make([]vk.DescriptorSetLayoutBinding, len(bindings))
  defer delete(vk_bindings)
  for &b, i in bindings {
    vk_bindings[i] = {
      binding         = u32(i),
      descriptorType  = b.type,
      descriptorCount = b.count,
      stageFlags      = b.flags,
    }
  }
  vk.CreateDescriptorSetLayout(
    gctx.device,
    &{
      sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      bindingCount = u32(len(vk_bindings)),
      pBindings = raw_data(vk_bindings),
    },
    nil,
    &layout,
  ) or_return
  ret = .SUCCESS
  return
}

create_pipeline_layout :: proc(
  gctx: ^GPUContext,
  pc_range: Maybe(vk.PushConstantRange),
  ds: ..vk.DescriptorSetLayout,
) -> (
  layout: vk.PipelineLayout,
  ret: vk.Result,
) {
  pc_range, has_pc := pc_range.?
  vk.CreatePipelineLayout(
    gctx.device,
    &vk.PipelineLayoutCreateInfo {
      sType = .PIPELINE_LAYOUT_CREATE_INFO,
      setLayoutCount = u32(len(ds)),
      pSetLayouts = raw_data(ds),
      pushConstantRangeCount = 1 if has_pc else 0,
      pPushConstantRanges = &pc_range,
    },
    nil,
    &layout,
  ) or_return
  ret = .SUCCESS
  return
}

bind_graphics_pipeline :: proc(
  command_buffer: vk.CommandBuffer,
  pipeline: vk.Pipeline,
  layout: vk.PipelineLayout,
  ds: ..vk.DescriptorSet,
) {
  vk.CmdBindPipeline(command_buffer, .GRAPHICS, pipeline)
  vk.CmdBindDescriptorSets(
    command_buffer,
    .GRAPHICS,
    layout,
    0,
    u32(len(ds)),
    raw_data(ds),
    0,
    nil,
  )
}

bind_compute_pipeline :: proc(
  command_buffer: vk.CommandBuffer,
  pipeline: vk.Pipeline,
  layout: vk.PipelineLayout,
  ds: ..vk.DescriptorSet,
) {
  vk.CmdBindPipeline(command_buffer, .COMPUTE, pipeline)
  vk.CmdBindDescriptorSets(
    command_buffer,
    .COMPUTE,
    layout,
    0,
    u32(len(ds)),
    raw_data(ds),
    0,
    nil,
  )
}

image_discard_barrier :: proc(
  command_buffer: vk.CommandBuffer,
  image: vk.Image,
  dst_access: vk.AccessFlags2,
  dst_stage: vk.PipelineStageFlags2,
  aspect_mask: vk.ImageAspectFlags,
  level_count: u32 = 1,
  layer_count: u32 = 1,
) {
  barrier := vk.ImageMemoryBarrier2 {
    sType = .IMAGE_MEMORY_BARRIER_2,
    srcStageMask = {.TOP_OF_PIPE},
    dstStageMask = dst_stage,
    dstAccessMask = dst_access,
    oldLayout = .UNDEFINED,
    newLayout = .GENERAL,
    srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
    dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
    image = image,
    subresourceRange = {
      aspectMask = aspect_mask,
      levelCount = level_count,
      layerCount = layer_count,
    },
  }
  dep := vk.DependencyInfo {
    sType                   = .DEPENDENCY_INFO,
    imageMemoryBarrierCount = 1,
    pImageMemoryBarriers    = &barrier,
  }
  vk.CmdPipelineBarrier2(command_buffer, &dep)
}

memory_barrier :: proc(
  command_buffer: vk.CommandBuffer,
  src_access: vk.AccessFlags2,
  dst_access: vk.AccessFlags2,
  src_stage: vk.PipelineStageFlags2,
  dst_stage: vk.PipelineStageFlags2,
) {
  barrier := vk.MemoryBarrier2 {
    sType         = .MEMORY_BARRIER_2,
    srcStageMask  = src_stage,
    srcAccessMask = src_access,
    dstStageMask  = dst_stage,
    dstAccessMask = dst_access,
  }
  dep := vk.DependencyInfo {
    sType              = .DEPENDENCY_INFO,
    memoryBarrierCount = 1,
    pMemoryBarriers    = &barrier,
  }
  vk.CmdPipelineBarrier2(command_buffer, &dep)
}

buffer_barrier :: proc(
  command_buffer: vk.CommandBuffer,
  buffer: vk.Buffer,
  size: vk.DeviceSize,
  src_access: vk.AccessFlags2,
  dst_access: vk.AccessFlags2,
  src_stage: vk.PipelineStageFlags2,
  dst_stage: vk.PipelineStageFlags2,
  offset: vk.DeviceSize = 0,
) {
  barrier := vk.BufferMemoryBarrier2 {
    sType               = .BUFFER_MEMORY_BARRIER_2,
    srcStageMask        = src_stage,
    srcAccessMask       = src_access,
    dstStageMask        = dst_stage,
    dstAccessMask       = dst_access,
    srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
    dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
    buffer              = buffer,
    offset              = offset,
    size                = size,
  }
  dep := vk.DependencyInfo {
    sType                    = .DEPENDENCY_INFO,
    bufferMemoryBarrierCount = 1,
    pBufferMemoryBarriers    = &barrier,
  }
  vk.CmdPipelineBarrier2(command_buffer, &dep)
}

create_compute_pipeline :: proc(
  gctx: ^GPUContext,
  shader: vk.ShaderModule,
  layout: vk.PipelineLayout,
  entry_point: cstring = "main",
) -> (
  pipeline: vk.Pipeline,
  ret: vk.Result,
) {
  info := vk.ComputePipelineCreateInfo {
    sType = .COMPUTE_PIPELINE_CREATE_INFO,
    stage = {
      sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage = {.COMPUTE},
      module = shader,
      pName = entry_point,
    },
    layout = layout,
  }
  vk.CreateComputePipelines(gctx.device, 0, 1, &info, nil, &pipeline) or_return
  ret = .SUCCESS
  return
}

bind_vertex_index_buffers :: proc(
  command_buffer: vk.CommandBuffer,
  vertex_buffer: vk.Buffer,
  index_buffer: vk.Buffer,
  vertex_offset: vk.DeviceSize = 0,
  index_offset: vk.DeviceSize = 0,
  index_type: vk.IndexType = .UINT32,
) {
  vertex_buffers := [?]vk.Buffer{vertex_buffer}
  offsets := [?]vk.DeviceSize{vertex_offset}
  vk.CmdBindVertexBuffers(
    command_buffer,
    0,
    1,
    raw_data(vertex_buffers[:]),
    raw_data(offsets[:]),
  )
  vk.CmdBindIndexBuffer(command_buffer, index_buffer, index_offset, index_type)
}

set_viewport_scissor :: proc(
  command_buffer: vk.CommandBuffer,
  extent: vk.Extent2D,
  flip_x: bool = false,
  flip_y: bool = true,
) {
  viewport := vk.Viewport {
    x        = flip_x ? f32(extent.width) : 0,
    y        = flip_y ? f32(extent.height) : 0,
    width    = flip_x ? -f32(extent.width) : f32(extent.width),
    height   = flip_y ? -f32(extent.height) : f32(extent.height),
    minDepth = 0.0,
    maxDepth = 1.0,
  }
  scissor := vk.Rect2D {
    extent = extent,
  }
  vk.CmdSetViewport(command_buffer, 0, 1, &viewport)
  vk.CmdSetScissor(command_buffer, 0, 1, &scissor)
}

begin_depth_rendering :: proc(
  command_buffer: vk.CommandBuffer,
  extent: vk.Extent2D,
  depth_attachment: ^vk.RenderingAttachmentInfo,
  layer_count: u32 = 1,
) {
  render_info := vk.RenderingInfo {
    sType = .RENDERING_INFO,
    renderArea = {extent = extent},
    layerCount = layer_count,
    pDepthAttachment = depth_attachment,
  }
  vk.CmdBeginRendering(command_buffer, &render_info)
}

create_color_attachment :: proc(
  image: ^Image,
  load_op: vk.AttachmentLoadOp = .CLEAR,
  store_op: vk.AttachmentStoreOp = .STORE,
  clear_color: [4]f32 = {0.0, 0.0, 0.0, 1.0},
) -> vk.RenderingAttachmentInfo {
  return vk.RenderingAttachmentInfo {
    sType = .RENDERING_ATTACHMENT_INFO,
    imageView = image.view,
    imageLayout = .GENERAL,
    loadOp = load_op,
    storeOp = store_op,
    clearValue = {color = {float32 = {0.0, 0.0, 0.0, 1.0}}},
  }
}

create_color_attachment_view :: proc(
  image_view: vk.ImageView,
  load_op: vk.AttachmentLoadOp = .CLEAR,
  store_op: vk.AttachmentStoreOp = .STORE,
  clear_color: [4]f32 = {0.0, 0.0, 0.0, 1.0},
) -> vk.RenderingAttachmentInfo {
  return vk.RenderingAttachmentInfo {
    sType = .RENDERING_ATTACHMENT_INFO,
    imageView = image_view,
    imageLayout = .GENERAL,
    loadOp = load_op,
    storeOp = store_op,
    clearValue = {color = {float32 = {0.0, 0.0, 0.0, 1.0}}},
  }
}

create_depth_attachment :: proc(
  image: ^Image,
  load_op: vk.AttachmentLoadOp = .CLEAR,
  store_op: vk.AttachmentStoreOp = .STORE,
  clear_depth: f32 = 1.0,
) -> vk.RenderingAttachmentInfo {
  return vk.RenderingAttachmentInfo {
    sType = .RENDERING_ATTACHMENT_INFO,
    imageView = image.view,
    imageLayout = .GENERAL,
    loadOp = load_op,
    storeOp = store_op,
    clearValue = {depthStencil = {depth = clear_depth}},
  }
}

create_cube_depth_attachment :: proc(
  image: ^CubeImage,
  load_op: vk.AttachmentLoadOp = .CLEAR,
  store_op: vk.AttachmentStoreOp = .STORE,
  clear_depth: f32 = 1.0,
) -> vk.RenderingAttachmentInfo {
  return vk.RenderingAttachmentInfo {
    sType = .RENDERING_ATTACHMENT_INFO,
    imageView = image.view,
    imageLayout = .GENERAL,
    loadOp = load_op,
    storeOp = store_op,
    clearValue = {depthStencil = {depth = clear_depth}},
  }
}

begin_rendering :: proc(
  command_buffer: vk.CommandBuffer,
  extent: vk.Extent2D,
  depth_attachment: Maybe(vk.RenderingAttachmentInfo),
  color_attachments: ..vk.RenderingAttachmentInfo,
) {
  depth, has_depth := depth_attachment.?
  render_info := vk.RenderingInfo {
    sType = .RENDERING_INFO,
    renderArea = {extent = extent},
    layerCount = 1,
    pDepthAttachment = &depth if has_depth else nil,
    colorAttachmentCount = u32(len(color_attachments)),
    pColorAttachments = raw_data(color_attachments),
  }
  vk.CmdBeginRendering(command_buffer, &render_info)
}
