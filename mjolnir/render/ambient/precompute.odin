package ambient

import "../../gpu"
import "core:log"
import vk "vendor:vulkan"

SHADER_EQUIRECT :: #load("../../shader/ibl/equirect_to_cube.spv")
SHADER_IRRADIANCE :: #load("../../shader/ibl/irradiance.spv")
SHADER_PREFILTER :: #load("../../shader/ibl/prefilter.spv")

ENV_CUBE_SIZE :: 512
IRRADIANCE_SIZE :: 32
PREFILTER_SIZE :: 128
PREFILTER_MIPS :: 5
PREFILTER_SAMPLES :: 1024
IRRADIANCE_SAMPLE_DELTA :: f32(0.025)
CUBE_FORMAT :: vk.Format.R16G16B16A16_SFLOAT

EquirectPush :: struct {
  face_size: u32,
}

IrradiancePush :: struct {
  face_size:    u32,
  sample_delta: f32,
}

PrefilterPush :: struct {
  face_size:     u32,
  roughness:     f32,
  sample_count:  u32,
  env_face_size: f32,
}

// Outputs from a single precompute run. The caller owns the handles and must
// free them via free_results.
IBLResults :: struct {
  env_cube:           gpu.TextureCubeHandle,
  irradiance_cube:    gpu.TextureCubeHandle,
  prefilter_cube:     gpu.TextureCubeHandle,
  prefilter_max_lod:  f32,
}

// Precompute IBL cubemaps from an equirectangular HDR source texture.
// Allocates 3 cube textures (env, irradiance, prefilter) and runs 3 compute
// passes via a one-shot command buffer. Safe to call once at engine setup.
precompute :: proc(
  gctx: ^gpu.GPUContext,
  texture_manager: ^gpu.TextureManager,
  equirect_handle: gpu.Texture2DHandle,
  linear_repeat_sampler: vk.Sampler,
) -> (
  results: IBLResults,
  ret: vk.Result,
) {
  equirect := gpu.get_texture_2d(texture_manager, equirect_handle)
  if equirect == nil {
    log.error("IBL precompute: equirect source texture is nil")
    return {}, .ERROR_UNKNOWN
  }

  // Allocate the three output cubemaps and register in bindless descriptor.
  results.env_cube = gpu.allocate_texture_cube_color(
    texture_manager,
    gctx,
    ENV_CUBE_SIZE,
    CUBE_FORMAT,
    1,
  ) or_return
  results.irradiance_cube = gpu.allocate_texture_cube_color(
    texture_manager,
    gctx,
    IRRADIANCE_SIZE,
    CUBE_FORMAT,
    1,
  ) or_return
  results.prefilter_cube = gpu.allocate_texture_cube_color(
    texture_manager,
    gctx,
    PREFILTER_SIZE,
    CUBE_FORMAT,
    PREFILTER_MIPS,
  ) or_return
  results.prefilter_max_lod = f32(PREFILTER_MIPS - 1)

  env_img := gpu.get_texture_cube(texture_manager, results.env_cube)
  irr_img := gpu.get_texture_cube(texture_manager, results.irradiance_cube)
  pre_img := gpu.get_texture_cube(texture_manager, results.prefilter_cube)

  // Per-mip 2D-array storage views: required because image2DArray writes
  // address all 6 faces of a *single* mip level at a time.
  env_storage_view := gpu.image_create_view(
    gctx.device,
    &env_img.base,
    .D2_ARRAY,
    0, 1, 0, 6,
  ) or_return
  defer vk.DestroyImageView(gctx.device, env_storage_view, nil)

  irr_storage_view := gpu.image_create_view(
    gctx.device,
    &irr_img.base,
    .D2_ARRAY,
    0, 1, 0, 6,
  ) or_return
  defer vk.DestroyImageView(gctx.device, irr_storage_view, nil)

  pre_storage_views: [PREFILTER_MIPS]vk.ImageView
  for i in 0 ..< u32(PREFILTER_MIPS) {
    pre_storage_views[i] = gpu.image_create_view(
      gctx.device,
      &pre_img.base,
      .D2_ARRAY,
      i, 1, 0, 6,
    ) or_return
  }
  defer for v in pre_storage_views do vk.DestroyImageView(gctx.device, v, nil)

  // Build pipelines (descriptor layouts: combined sampler + storage image).
  ds_layout, layout_ret := create_ds_layout(gctx)
  if layout_ret != .SUCCESS do return results, layout_ret
  defer vk.DestroyDescriptorSetLayout(gctx.device, ds_layout, nil)

  equirect_layout := gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange{stageFlags = {.COMPUTE}, size = size_of(EquirectPush)},
    ds_layout,
  ) or_return
  defer vk.DestroyPipelineLayout(gctx.device, equirect_layout, nil)

  irradiance_layout := gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange{stageFlags = {.COMPUTE}, size = size_of(IrradiancePush)},
    ds_layout,
  ) or_return
  defer vk.DestroyPipelineLayout(gctx.device, irradiance_layout, nil)

  prefilter_layout := gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange{stageFlags = {.COMPUTE}, size = size_of(PrefilterPush)},
    ds_layout,
  ) or_return
  defer vk.DestroyPipelineLayout(gctx.device, prefilter_layout, nil)

  equirect_mod := gpu.create_shader_module(gctx.device, SHADER_EQUIRECT) or_return
  defer vk.DestroyShaderModule(gctx.device, equirect_mod, nil)
  irradiance_mod := gpu.create_shader_module(gctx.device, SHADER_IRRADIANCE) or_return
  defer vk.DestroyShaderModule(gctx.device, irradiance_mod, nil)
  prefilter_mod := gpu.create_shader_module(gctx.device, SHADER_PREFILTER) or_return
  defer vk.DestroyShaderModule(gctx.device, prefilter_mod, nil)

  equirect_pipe := gpu.create_compute_pipeline(gctx, equirect_mod, equirect_layout) or_return
  defer vk.DestroyPipeline(gctx.device, equirect_pipe, nil)
  irradiance_pipe := gpu.create_compute_pipeline(gctx, irradiance_mod, irradiance_layout) or_return
  defer vk.DestroyPipeline(gctx.device, irradiance_pipe, nil)
  prefilter_pipe := gpu.create_compute_pipeline(gctx, prefilter_mod, prefilter_layout) or_return
  defer vk.DestroyPipeline(gctx.device, prefilter_pipe, nil)

  // One descriptor set per dispatch (equirect, irradiance, + N for prefilter mips).
  // Prefilter samples from the env cubemap (read via env_img.view) but writes
  // to per-mip views; we allocate PREFILTER_MIPS sets so writes don't alias.
  total_sets := 2 + PREFILTER_MIPS
  ds: [2 + PREFILTER_MIPS]vk.DescriptorSet
  ds_slice := ds[:total_sets]
  gpu.allocate_descriptor_set(gctx, ds_slice, ds_layout) or_return

  // ds[0]: equirect sampler2D -> env cube storage
  write_compute_ds(
    gctx,
    ds[0],
    linear_repeat_sampler,
    equirect.view,
    env_storage_view,
  )
  // ds[1]: env cube sampler -> irradiance cube storage
  write_compute_ds(
    gctx,
    ds[1],
    linear_repeat_sampler,
    env_img.view,
    irr_storage_view,
  )
  // ds[2..]: env cube sampler -> prefilter mip i storage
  for i in 0 ..< PREFILTER_MIPS {
    write_compute_ds(
      gctx,
      ds[2 + i],
      linear_repeat_sampler,
      env_img.view,
      pre_storage_views[i],
    )
  }

  cmd := gpu.begin_single_time_command(gctx) or_return

  // 1. Equirect 2D -> env cubemap.
  // env cube was just allocated; move out of UNDEFINED for storage write.
  gpu.image_discard_barrier(
    cmd,
    env_img.image,
    {.SHADER_WRITE},
    {.COMPUTE_SHADER},
    {.COLOR},
    layer_count = 6,
  )
  vk.CmdBindPipeline(cmd, .COMPUTE, equirect_pipe)
  vk.CmdBindDescriptorSets(cmd, .COMPUTE, equirect_layout, 0, 1, &ds[0], 0, nil)
  ep := EquirectPush{face_size = ENV_CUBE_SIZE}
  vk.CmdPushConstants(
    cmd, equirect_layout, {.COMPUTE}, 0, size_of(EquirectPush), &ep,
  )
  vk.CmdDispatch(cmd, (ENV_CUBE_SIZE + 7) / 8, (ENV_CUBE_SIZE + 7) / 8, 6)

  // Make env cubemap writes visible to downstream sampling.
  gpu.memory_barrier(
    cmd,
    {.SHADER_WRITE}, {.SHADER_READ},
    {.COMPUTE_SHADER}, {.COMPUTE_SHADER},
  )

  // 2. Env cube -> irradiance cubemap.
  gpu.image_discard_barrier(
    cmd,
    irr_img.image,
    {.SHADER_WRITE},
    {.COMPUTE_SHADER},
    {.COLOR},
    layer_count = 6,
  )
  vk.CmdBindPipeline(cmd, .COMPUTE, irradiance_pipe)
  vk.CmdBindDescriptorSets(cmd, .COMPUTE, irradiance_layout, 0, 1, &ds[1], 0, nil)
  ip := IrradiancePush{
    face_size = IRRADIANCE_SIZE,
    sample_delta = IRRADIANCE_SAMPLE_DELTA,
  }
  vk.CmdPushConstants(
    cmd, irradiance_layout, {.COMPUTE}, 0, size_of(IrradiancePush), &ip,
  )
  vk.CmdDispatch(cmd, (IRRADIANCE_SIZE + 7) / 8, (IRRADIANCE_SIZE + 7) / 8, 6)
  gpu.memory_barrier(
    cmd,
    {.SHADER_WRITE}, {.SHADER_READ},
    {.COMPUTE_SHADER}, {.FRAGMENT_SHADER},
  )

  // 3. Env cube -> prefiltered specular cube (per-mip).
  gpu.image_discard_barrier(
    cmd,
    pre_img.image,
    {.SHADER_WRITE},
    {.COMPUTE_SHADER},
    {.COLOR},
    level_count = PREFILTER_MIPS,
    layer_count = 6,
  )
  vk.CmdBindPipeline(cmd, .COMPUTE, prefilter_pipe)
  for i in 0 ..< u32(PREFILTER_MIPS) {
    mip_size := u32(max(PREFILTER_SIZE >> i, 1))
    roughness := f32(i) / f32(PREFILTER_MIPS - 1)
    pp := PrefilterPush{
      face_size     = mip_size,
      roughness     = roughness,
      sample_count  = PREFILTER_SAMPLES,
      env_face_size = ENV_CUBE_SIZE,
    }
    ds_ptr := &ds[2 + i]
    vk.CmdBindDescriptorSets(cmd, .COMPUTE, prefilter_layout, 0, 1, ds_ptr, 0, nil)
    vk.CmdPushConstants(
      cmd, prefilter_layout, {.COMPUTE}, 0, size_of(PrefilterPush), &pp,
    )
    vk.CmdDispatch(cmd, (mip_size + 7) / 8, (mip_size + 7) / 8, 6)
  }
  gpu.memory_barrier(
    cmd,
    {.SHADER_WRITE}, {.SHADER_READ},
    {.COMPUTE_SHADER}, {.FRAGMENT_SHADER},
  )

  gpu.end_single_time_command(gctx, &cmd) or_return
  log.infof(
    "IBL precompute done: env=%d^2, irradiance=%d^2, prefilter=%d^2 x %d mips",
    ENV_CUBE_SIZE, IRRADIANCE_SIZE, PREFILTER_SIZE, PREFILTER_MIPS,
  )
  return results, .SUCCESS
}

free_results :: proc(
  gctx: ^gpu.GPUContext,
  texture_manager: ^gpu.TextureManager,
  r: ^IBLResults,
) {
  gpu.free_texture_cube(texture_manager, gctx, r.env_cube)
  gpu.free_texture_cube(texture_manager, gctx, r.irradiance_cube)
  gpu.free_texture_cube(texture_manager, gctx, r.prefilter_cube)
  r^ = {}
}

@(private)
create_ds_layout :: proc(
  gctx: ^gpu.GPUContext,
) -> (
  layout: vk.DescriptorSetLayout,
  ret: vk.Result,
) {
  return gpu.create_descriptor_set_layout(
    gctx,
    {type = .COMBINED_IMAGE_SAMPLER, flags = {.COMPUTE}},
    {type = .STORAGE_IMAGE, flags = {.COMPUTE}},
  )
}

@(private)
write_compute_ds :: proc(
  gctx: ^gpu.GPUContext,
  dst: vk.DescriptorSet,
  sampler: vk.Sampler,
  src_view: vk.ImageView,
  dst_view: vk.ImageView,
) {
  gpu.update_descriptor_set(
    gctx,
    dst,
    {
      type = .COMBINED_IMAGE_SAMPLER,
      info = vk.DescriptorImageInfo{
        sampler = sampler,
        imageView = src_view,
        imageLayout = .GENERAL,
      },
    },
    {
      type = .STORAGE_IMAGE,
      info = vk.DescriptorImageInfo{
        imageView = dst_view,
        imageLayout = .GENERAL,
      },
    },
  )
}
