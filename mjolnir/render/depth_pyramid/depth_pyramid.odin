package depth_pyramid

import alg "../../algebra"
import "../../gpu"
import "core:log"
import "core:math"
import vk "vendor:vulkan"

SHADER_DEPTH_REDUCE :: #load("../../shader/occlusion_culling/depth_reduce.spv")
MAX_DEPTH_MIPS_LEVEL :: 16
FRAMES_IN_FLIGHT :: 2

DepthReducePushConstants :: struct {
  current_mip: u32,
}

System :: struct {
  depth_reduce_layout:            vk.PipelineLayout,
  depth_reduce_pipeline:          vk.Pipeline,
  depth_reduce_descriptor_layout: vk.DescriptorSetLayout,
}

// DepthPyramid - Hierarchical depth buffer for occlusion culling (GPU resource)
DepthPyramid :: struct {
  texture:      gpu.Texture2DHandle,
  views:        [MAX_DEPTH_MIPS_LEVEL]vk.ImageView,
  full_view:    vk.ImageView,
  sampler:      vk.Sampler,
  mip_levels:   u32,
  using extent: vk.Extent2D,
}

init :: proc(
  self: ^System,
  gctx: ^gpu.GPUContext,
) -> (
  ret: vk.Result,
) {
  self.depth_reduce_descriptor_layout = gpu.create_descriptor_set_layout(
    gctx,
    {.COMBINED_IMAGE_SAMPLER, {.COMPUTE}},
    {.STORAGE_IMAGE, {.COMPUTE}},
  ) or_return
  defer if ret != .SUCCESS {
    vk.DestroyDescriptorSetLayout(
      gctx.device,
      self.depth_reduce_descriptor_layout,
      nil,
    )
    self.depth_reduce_descriptor_layout = 0
  }
  self.depth_reduce_layout = gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange {
      stageFlags = {.COMPUTE},
      size = size_of(DepthReducePushConstants),
    },
    self.depth_reduce_descriptor_layout,
  ) or_return
  defer if ret != .SUCCESS {
    vk.DestroyPipelineLayout(gctx.device, self.depth_reduce_layout, nil)
    self.depth_reduce_layout = 0
  }
  depth_shader := gpu.create_shader_module(
    gctx.device,
    SHADER_DEPTH_REDUCE,
  ) or_return
  defer vk.DestroyShaderModule(gctx.device, depth_shader, nil)
  self.depth_reduce_pipeline = gpu.create_compute_pipeline(
    gctx,
    depth_shader,
    self.depth_reduce_layout,
  ) or_return
  return .SUCCESS
}

shutdown :: proc(self: ^System, gctx: ^gpu.GPUContext) {
  vk.DestroyPipeline(gctx.device, self.depth_reduce_pipeline, nil)
  self.depth_reduce_pipeline = 0
  vk.DestroyPipelineLayout(gctx.device, self.depth_reduce_layout, nil)
  self.depth_reduce_layout = 0
  vk.DestroyDescriptorSetLayout(
    gctx.device,
    self.depth_reduce_descriptor_layout,
    nil,
  )
  self.depth_reduce_descriptor_layout = 0
}

// Create depth pyramid texture and views
setup_pyramid :: proc(
  gctx: ^gpu.GPUContext,
  pyramid: ^DepthPyramid,
  texture_manager: ^gpu.TextureManager,
  extent: vk.Extent2D,
) -> vk.Result {
  extent := vk.Extent2D{max(1, extent.width / 2), max(1, extent.height / 2)}
  mip_levels := alg.log2_greater_than(max(extent.width, extent.height))
  pyramid_handle := gpu.allocate_texture_2d(
    texture_manager,
    gctx,
    extent,
    .R32_SFLOAT,
    {.SAMPLED, .STORAGE, .TRANSFER_DST},
    true, // generate_mips
  ) or_return

  pyramid_texture := gpu.get_texture_2d(texture_manager, pyramid_handle)
  if pyramid_texture == nil {
    log.error("Failed to get allocated depth pyramid texture")
    return .ERROR_OUT_OF_DEVICE_MEMORY
  }

  // Move all mip levels out of UNDEFINED
  {
    cmd_buf := gpu.begin_single_time_command(gctx) or_return
    gpu.image_discard_barrier(
      cmd_buf,
      pyramid_texture.image,
      {.SHADER_READ, .SHADER_WRITE},
      {.COMPUTE_SHADER},
      {.COLOR},
      level_count = mip_levels,
    )
    gpu.end_single_time_command(gctx, &cmd_buf) or_return
  }

  pyramid.texture = pyramid_handle
  pyramid.mip_levels = mip_levels
  pyramid.extent = extent

  // Create per-mip views
  for mip in 0 ..< mip_levels {
    view_info := vk.ImageViewCreateInfo {
      sType = .IMAGE_VIEW_CREATE_INFO,
      image = pyramid_texture.image,
      viewType = .D2,
      format = .R32_SFLOAT,
      subresourceRange = {
        aspectMask = {.COLOR},
        baseMipLevel = mip,
        levelCount = 1,
        layerCount = 1,
      },
    }
    vk.CreateImageView(
      gctx.device,
      &view_info,
      nil,
      &pyramid.views[mip],
    ) or_return
  }

  // Create full pyramid view
  full_view_info := vk.ImageViewCreateInfo {
    sType = .IMAGE_VIEW_CREATE_INFO,
    image = pyramid_texture.image,
    viewType = .D2,
    format = .R32_SFLOAT,
    subresourceRange = {
      aspectMask = {.COLOR},
      baseMipLevel = 0,
      levelCount = mip_levels,
      layerCount = 1,
    },
  }
  vk.CreateImageView(
    gctx.device,
    &full_view_info,
    nil,
    &pyramid.full_view,
  ) or_return

  // Create sampler for depth pyramid with MAX reduction for forward-Z
  reduction_mode := vk.SamplerReductionModeCreateInfo {
    sType         = .SAMPLER_REDUCTION_MODE_CREATE_INFO,
    reductionMode = .MAX,
  }
  sampler_info := vk.SamplerCreateInfo {
    sType        = .SAMPLER_CREATE_INFO,
    magFilter    = .LINEAR,
    minFilter    = .LINEAR,
    mipmapMode   = .NEAREST,
    addressModeU = .CLAMP_TO_EDGE,
    addressModeV = .CLAMP_TO_EDGE,
    addressModeW = .CLAMP_TO_EDGE,
    minLod       = 0.0,
    maxLod       = f32(mip_levels),
    borderColor  = .FLOAT_OPAQUE_WHITE,
    pNext        = &reduction_mode,
  }
  vk.CreateSampler(
    gctx.device,
    &sampler_info,
    nil,
    &pyramid.sampler,
  ) or_return

  return .SUCCESS
}

// Destroy depth pyramid resources
destroy_pyramid :: proc(
  gctx: ^gpu.GPUContext,
  pyramid: ^DepthPyramid,
  texture_manager: ^gpu.TextureManager,
) {
  if pyramid.mip_levels == 0 do return

  for mip in 0 ..< pyramid.mip_levels {
    vk.DestroyImageView(gctx.device, pyramid.views[mip], nil)
  }
  vk.DestroyImageView(gctx.device, pyramid.full_view, nil)
  vk.DestroySampler(gctx.device, pyramid.sampler, nil)

  gpu.free_texture_2d(texture_manager, gctx, pyramid.texture)

  pyramid^ = {}
}

// Build depth pyramid from depth buffer
build_pyramid :: proc(
  self: ^System,
  command_buffer: vk.CommandBuffer,
  node_count: u32,
  pyramid: ^DepthPyramid,
  descriptor_sets: []vk.DescriptorSet,
) {
  if node_count == 0 do return
  if pyramid.mip_levels == 0 do return

  vk.CmdBindPipeline(command_buffer, .COMPUTE, self.depth_reduce_pipeline)
  for mip in 0 ..< pyramid.mip_levels {
    vk.CmdBindDescriptorSets(
      command_buffer,
      .COMPUTE,
      self.depth_reduce_layout,
      0,
      1,
      &descriptor_sets[mip],
      0,
      nil,
    )
    push_constants := DepthReducePushConstants {
      current_mip = u32(mip),
    }
    vk.CmdPushConstants(
      command_buffer,
      self.depth_reduce_layout,
      {.COMPUTE},
      0,
      size_of(push_constants),
      &push_constants,
    )
    mip_width := max(1, pyramid.width >> mip)
    mip_height := max(1, pyramid.height >> mip)
    dispatch_x := (mip_width + 31) / 32
    dispatch_y := (mip_height + 31) / 32
    vk.CmdDispatch(command_buffer, dispatch_x, dispatch_y, 1)
    if mip < pyramid.mip_levels - 1 {
      gpu.memory_barrier(
        command_buffer,
        {.SHADER_WRITE},
        {.SHADER_READ},
        {.COMPUTE_SHADER},
        {.COMPUTE_SHADER},
      )
    }
  }
}
