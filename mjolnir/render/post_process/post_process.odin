package post_process

import cont "../../containers"
import "../../gpu"
import "../shared"
import "core:log"
import vk "vendor:vulkan"

SHADER_POSTPROCESS_VERT :: #load("../../shader/postprocess/vert.spv")
SHADER_POSTPROCESS_FRAG :: #load("../../shader/postprocess/frag.spv")
SHADER_BLOOM_FRAG :: #load("../../shader/bloom/frag.spv")
SHADER_BLUR_FRAG :: #load("../../shader/blur/frag.spv")
SHADER_GRAYSCALE_FRAG :: #load("../../shader/grayscale/frag.spv")
SHADER_TONEMAP_FRAG :: #load("../../shader/tonemap/frag.spv")
SHADER_OUTLINE_FRAG :: #load("../../shader/outline/frag.spv")
SHADER_FOG_FRAG :: #load("../../shader/fog/frag.spv")
SHADER_CROSSHATCH_FRAG :: #load("../../shader/crosshatch/frag.spv")
SHADER_DOF_FRAG :: #load("../../shader/dof/frag.spv")

BG_BLUE_GRAY :: [4]f32{0.0117, 0.0117, 0.0179, 1.0}
BG_DARK_GRAY :: [4]f32{0.0117, 0.0117, 0.0117, 1.0}
BG_ORANGE_GRAY :: [4]f32{0.0179, 0.0179, 0.0117, 1.0}

GrayscaleEffect :: struct {
  weights:  [3]f32,
  strength: f32,
}

ToneMapEffect :: struct {
  exposure: f32,
  gamma:    f32,
  padding:  [2]f32,
}

BlurEffect :: struct {
  radius:         f32,
  direction:      f32, // 0.0 = horizontal, 1.0 = vertical
  weight_falloff: f32, // 0.0 = box blur, 1.0 = gaussian blur
  padding:        f32,
}

BloomEffect :: struct {
  threshold:   f32,
  intensity:   f32,
  blur_radius: f32,
  direction:   f32, // 0.0 = horizontal, 1.0 = vertical
}

OutlineEffect :: struct {
  color:     [3]f32,
  thickness: f32,
}

FogEffect :: struct {
  color:   [3]f32,
  density: f32,
  start:   f32,
  end:     f32,
}

CrossHatchEffect :: struct {
  resolution:       [2]f32,
  hatch_offset_y:   f32,
  lum_threshold_01: f32,
  lum_threshold_02: f32,
  lum_threshold_03: f32,
  lum_threshold_04: f32,
  padding:          f32,
}

DoFEffect :: struct {
  focus_distance:  f32, // Distance to the focus plane
  focus_range:     f32, // Range where objects are in focus
  blur_strength:   f32, // Maximum blur radius
  bokeh_intensity: f32, // Bokeh effect intensity
}

PostProcessEffectType :: enum int {
  GRAYSCALE,
  TONEMAP,
  BLUR,
  BLOOM,
  OUTLINE,
  FOG,
  CROSSHATCH,
  DOF,
  NONE,
}

// Combined push constant structures for each effect type
GrayscalePushConstant :: struct {
  using base: BasePushConstant,
  // Effect parameters
  weights:    [3]f32,
  strength:   f32,
}

ToneMapPushConstant :: struct {
  using base: BasePushConstant,
  // Effect parameters
  exposure:   f32,
  gamma:      f32,
}

BlurPushConstant :: struct {
  using base:     BasePushConstant,
  // Effect parameters
  radius:         f32,
  direction:      f32,
  weight_falloff: f32,
}

BloomPushConstant :: struct {
  using base:  BasePushConstant,
  // Effect parameters
  threshold:   f32,
  intensity:   f32,
  blur_radius: f32,
  direction:   f32,
}

OutlinePushConstant :: struct {
  using base: BasePushConstant,
  // Effect parameters
  color:      [3]f32,
  thickness:  f32,
}

FogPushConstant :: struct {
  using base: BasePushConstant,
  // Effect parameters
  color:      [4]f32, // Changed from [3]f32 to [4]f32 to match GLSL vec4
  density:    f32,
  start:      f32,
  end:        f32,
}

CrossHatchPushConstant :: struct {
  using base:       BasePushConstant,
  // Effect parameters
  resolution:       [2]f32,
  hatch_offset_y:   f32,
  lum_threshold_01: f32,
  lum_threshold_02: f32,
  lum_threshold_03: f32,
  lum_threshold_04: f32,
}

DoFPushConstant :: struct {
  using base:      BasePushConstant,
  // Effect parameters
  focus_distance:  f32,
  focus_range:     f32,
  blur_strength:   f32,
  bokeh_intensity: f32,
}

BasePushConstant :: struct {
  position_texture_index: u32,
  normal_texture_index:   u32,
  albedo_texture_index:   u32,
  metallic_texture_index: u32,
  emissive_texture_index: u32,
  depth_texture_index:    u32,
  input_image_index:      u32,
  padding:                u32, // Add padding to align to 16-byte boundary for next vec4
}

PostprocessEffect :: union {
  GrayscaleEffect,
  ToneMapEffect,
  BlurEffect,
  BloomEffect,
  OutlineEffect,
  FogEffect,
  CrossHatchEffect,
  DoFEffect,
}

Renderer :: struct {
  pipelines_hdr:    [len(PostProcessEffectType)]vk.Pipeline,
  pipelines_ldr:    [len(PostProcessEffectType)]vk.Pipeline,
  pipeline_layouts: [len(PostProcessEffectType)]vk.PipelineLayout,
  effect_stack:     [dynamic]PostprocessEffect,
  // 3 ping-pong buffers: bloom V pass needs to read H output AND pre-bloom
  // scene simultaneously while writing to a third slot. With only 2 buffers
  // V would have to write to one of its inputs when bloom isn't the last pass.
  images:           [3]gpu.Texture2DHandle,
}

get_effect_type :: proc(effect: PostprocessEffect) -> PostProcessEffectType {
  switch _ in effect {
  case GrayscaleEffect:
    return .GRAYSCALE
  case ToneMapEffect:
    return .TONEMAP
  case BlurEffect:
    return .BLUR
  case BloomEffect:
    return .BLOOM
  case OutlineEffect:
    return .OUTLINE
  case FogEffect:
    return .FOG
  case CrossHatchEffect:
    return .CROSSHATCH
  case DoFEffect:
    return .DOF
  }
  return .NONE
}

add_grayscale :: proc(
  self: ^Renderer,
  strength: f32 = 1.0,
  weights: [3]f32 = {0.299, 0.587, 0.114},
) {
  effect := GrayscaleEffect {
    strength = strength,
    weights  = weights,
  }
  append(&self.effect_stack, effect)
}

add_blur :: proc(self: ^Renderer, radius: f32, gaussian: bool = true) {
  horizontal_effect := BlurEffect {
    radius         = radius,
    direction      = 0.0, // horizontal
    weight_falloff = 1.0 if gaussian else 0.0,
  }
  append(&self.effect_stack, horizontal_effect)
  vertical_effect := BlurEffect {
    radius         = radius,
    direction      = 1.0, // vertical
    weight_falloff = 1.0 if gaussian else 0.0,
  }
  append(&self.effect_stack, vertical_effect)
}

add_directional_blur :: proc(
  self: ^Renderer,
  radius: f32,
  direction: f32 = 0.0, // 0.0 = horizontal, 1.0 = vertical
  gaussian: bool = true,
) {
  effect := BlurEffect {
    radius         = radius,
    direction      = direction,
    weight_falloff = 1.0 if gaussian else 0.0,
  }
  append(&self.effect_stack, effect)
}

add_bloom :: proc(
  self: ^Renderer,
  threshold: f32 = 0.2,
  intensity: f32 = 1.0,
  blur_radius: f32 = 4.0,
) {
  effect := BloomEffect {
    threshold   = threshold,
    intensity   = intensity,
    blur_radius = blur_radius,
    direction   = 0.0, // horizontal
  }
  append(&self.effect_stack, effect)
  effect = BloomEffect {
    threshold   = threshold,
    intensity   = intensity,
    blur_radius = blur_radius,
    direction   = 1.0, // vertical
  }
  append(&self.effect_stack, effect)
}

add_tonemap :: proc(self: ^Renderer, exposure: f32 = 1.0, gamma: f32 = 1.0) {
  effect := ToneMapEffect {
    exposure = exposure,
    gamma    = gamma,
  }
  append(&self.effect_stack, effect)
}

add_outline :: proc(self: ^Renderer, thickness: f32, color: [3]f32 = {0, 0, 0}) {
  effect := OutlineEffect {
    thickness = thickness,
    color     = color,
  }
  append(&self.effect_stack, effect)
}

add_fog :: proc(
  self: ^Renderer,
  color: [3]f32 = {0.7, 0.7, 0.8},
  density: f32 = 0.02,
  start: f32 = 10.0,
  end: f32 = 100.0,
) {
  effect := FogEffect {
    color   = color,
    density = density,
    start   = start,
    end     = end,
  }
  append(&self.effect_stack, effect)
}

add_crosshatch :: proc(
  self: ^Renderer,
  resolution: [2]f32,
  hatch_offset_y: f32 = 5.0,
  lum_threshold_01: f32 = 0.6,
  lum_threshold_02: f32 = 0.3,
  lum_threshold_03: f32 = 0.15,
  lum_threshold_04: f32 = 0.07,
) {
  effect := CrossHatchEffect {
    resolution       = resolution,
    hatch_offset_y   = hatch_offset_y,
    lum_threshold_01 = lum_threshold_01,
    lum_threshold_02 = lum_threshold_02,
    lum_threshold_03 = lum_threshold_03,
    lum_threshold_04 = lum_threshold_04,
  }
  append(&self.effect_stack, effect)
}

add_dof :: proc(
  self: ^Renderer,
  focus_distance: f32 = 3.0,
  focus_range: f32 = 2.0,
  blur_strength: f32 = 20.0,
  bokeh_intensity: f32 = 0.5,
) {
  effect := DoFEffect {
    focus_distance  = focus_distance,
    focus_range     = focus_range,
    blur_strength   = blur_strength,
    bokeh_intensity = bokeh_intensity,
  }
  append(&self.effect_stack, effect)
}

clear_effects :: proc(self: ^Renderer) {
  clear(&self.effect_stack)
}

effect_clear :: proc(self: ^Renderer) {
  clear(&self.effect_stack)
}

setup :: proc(
  self: ^Renderer,
  gctx: ^gpu.GPUContext,
  texture_manager: ^gpu.TextureManager,
  extent: vk.Extent2D,
  format: vk.Format,
) -> vk.Result {
  return create_images(gctx, self, texture_manager, extent, format)
}

teardown :: proc(
  self: ^Renderer,
  gctx: ^gpu.GPUContext,
  texture_manager: ^gpu.TextureManager,
) {
  destroy_images(self, gctx, texture_manager)
}

init :: proc(
  self: ^Renderer,
  gctx: ^gpu.GPUContext,
  color_format: vk.Format,
  textures_set_layout: vk.DescriptorSetLayout,
) -> (
  ret: vk.Result,
) {
  self.effect_stack = make([dynamic]PostprocessEffect)
  defer if ret != .SUCCESS do delete(self.effect_stack)
  count :: len(PostProcessEffectType)
  vert_module := gpu.create_shader_module(
    gctx.device,
    SHADER_POSTPROCESS_VERT,
  ) or_return
  defer vk.DestroyShaderModule(gctx.device, vert_module, nil)
  frag_modules: [count]vk.ShaderModule
  defer for m in frag_modules do vk.DestroyShaderModule(gctx.device, m, nil)
  for effect_type, i in PostProcessEffectType {
    shader_code: []u8
    switch effect_type {
    case .BLOOM:
      shader_code = SHADER_BLOOM_FRAG
    case .BLUR:
      shader_code = SHADER_BLUR_FRAG
    case .GRAYSCALE:
      shader_code = SHADER_GRAYSCALE_FRAG
    case .TONEMAP:
      shader_code = SHADER_TONEMAP_FRAG
    case .OUTLINE:
      shader_code = SHADER_OUTLINE_FRAG
    case .FOG:
      shader_code = SHADER_FOG_FRAG
    case .CROSSHATCH:
      shader_code = SHADER_CROSSHATCH_FRAG
    case .DOF:
      shader_code = SHADER_DOF_FRAG
    case .NONE:
      shader_code = SHADER_POSTPROCESS_FRAG
    }
    frag_modules[i] = gpu.create_shader_module(
      gctx.device,
      shader_code,
    ) or_return
  }
  shader_stages: [count][2]vk.PipelineShaderStageCreateInfo
  hdr_infos: [count]vk.GraphicsPipelineCreateInfo
  ldr_infos: [count]vk.GraphicsPipelineCreateInfo
  ldr_format := color_format
  ldr_rendering_info := vk.PipelineRenderingCreateInfo {
    sType                   = .PIPELINE_RENDERING_CREATE_INFO,
    colorAttachmentCount    = 1,
    pColorAttachmentFormats = &ldr_format,
  }
  for effect_type, i in PostProcessEffectType {
    push_constant_size: u32
    switch effect_type {
    case .BLUR:
      push_constant_size = size_of(BlurPushConstant)
    case .OUTLINE:
      push_constant_size = size_of(OutlinePushConstant)
    case .GRAYSCALE:
      push_constant_size = size_of(GrayscalePushConstant)
    case .BLOOM:
      push_constant_size = size_of(BloomPushConstant)
    case .TONEMAP:
      push_constant_size = size_of(ToneMapPushConstant)
    case .FOG:
      push_constant_size = size_of(FogPushConstant)
    case .CROSSHATCH:
      push_constant_size = size_of(CrossHatchPushConstant)
    case .DOF:
      push_constant_size = size_of(DoFPushConstant)
    case .NONE:
      push_constant_size = size_of(BasePushConstant)
    }
    self.pipeline_layouts[i] = gpu.create_pipeline_layout(
      gctx,
      vk.PushConstantRange{stageFlags = {.FRAGMENT}, size = push_constant_size} if push_constant_size > 0 else nil,
      textures_set_layout,
    ) or_return
    shader_stages[i] = gpu.create_vert_frag_stages(
      vert_module,
      frag_modules[i],
      &shared.SHADER_SPEC_CONSTANTS,
    )
    hdr_infos[i] = {
      sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
      pNext               = &gpu.HDR_COLOR_ONLY_RENDERING_INFO,
      stageCount          = len(shader_stages[i]),
      pStages             = raw_data(shader_stages[i][:]),
      pVertexInputState   = &gpu.VERTEX_INPUT_NONE,
      pInputAssemblyState = &gpu.STANDARD_INPUT_ASSEMBLY,
      pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
      pRasterizationState = &gpu.DOUBLE_SIDED_RASTERIZER,
      pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
      pColorBlendState    = &gpu.COLOR_BLENDING_OVERRIDE,
      pDynamicState       = &gpu.STANDARD_DYNAMIC_STATES,
      layout              = self.pipeline_layouts[i],
    }
    ldr_infos[i] = hdr_infos[i]
    ldr_infos[i].pNext = &ldr_rendering_info
  }
  vk.CreateGraphicsPipelines(
    gctx.device,
    0,
    count,
    raw_data(hdr_infos[:]),
    nil,
    raw_data(self.pipelines_hdr[:]),
  ) or_return
  vk.CreateGraphicsPipelines(
    gctx.device,
    0,
    count,
    raw_data(ldr_infos[:]),
    nil,
    raw_data(self.pipelines_ldr[:]),
  ) or_return
  log.info("Postprocess pipeline initialized successfully")
  return .SUCCESS
}

create_images :: proc(
  gctx: ^gpu.GPUContext,
  self: ^Renderer,
  texture_manager: ^gpu.TextureManager,
  extent: vk.Extent2D,
  format: vk.Format,
) -> vk.Result {
  for &handle in self.images {
    handle = gpu.allocate_texture_2d(
      texture_manager,
      gctx,
      extent,
      gpu.HDR_COLOR_FORMAT,
      vk.ImageUsageFlags {
        .COLOR_ATTACHMENT,
        .SAMPLED,
        .TRANSFER_SRC,
        .TRANSFER_DST,
      },
    ) or_return

  }
  log.debugf("created post-process image")
  return .SUCCESS
}

destroy_images :: proc(
  self: ^Renderer,
  gctx: ^gpu.GPUContext,
  texture_manager: ^gpu.TextureManager,
) {
  for handle in self.images {
    gpu.free_texture_2d(texture_manager, gctx, handle)
  }
}

recreate_images :: proc(
  gctx: ^gpu.GPUContext,
  self: ^Renderer,
  texture_manager: ^gpu.TextureManager,
  extent: vk.Extent2D,
  format: vk.Format,
) -> vk.Result {
  destroy_images(self, gctx, texture_manager)
  return create_images(gctx, self, texture_manager, extent, format)
}

shutdown :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) {
  for &p in self.pipelines_hdr {
    vk.DestroyPipeline(gctx.device, p, nil)
    p = 0
  }
  for &p in self.pipelines_ldr {
    vk.DestroyPipeline(gctx.device, p, nil)
    p = 0
  }
  for &layout in self.pipeline_layouts {
    vk.DestroyPipelineLayout(gctx.device, layout, nil)
    layout = 0
  }
  delete(self.effect_stack)
}

// record runs the post-process effect stack into output_view. If the stack is
// empty a single copy effect is appended so the input always reaches output.
record :: proc(
  self: ^Renderer,
  command_buffer: vk.CommandBuffer,
  extent: vk.Extent2D,
  output_view: vk.ImageView,
  final_image_idx: u32,
  position_texture_idx: u32,
  normal_texture_idx: u32,
  albedo_texture_idx: u32,
  metallic_texture_idx: u32,
  emissive_texture_idx: u32,
  depth_texture_idx: u32,
  texture_manager: ^gpu.TextureManager,
) {
  if len(self.effect_stack) == 0 {
    // if no postprocess effect, just copy the input to output
    append(&self.effect_stack, nil)
  }
  gpu.set_viewport_scissor(command_buffer, extent, flip_y = false)
  // Slot tracking for 3-buffer ping-pong.
  //   current_input_slot == -1 means current input is `final_image_idx`
  //   (external, not one of self.images), so no transition required.
  //   bloom_h_input_slot is set when an H pass runs and consumed by the V pass
  //   that follows, so V can composite against the scene as it entered bloom.
  current_input_slot := -1
  current_input_image_idx := final_image_idx
  bloom_h_input_slot := -1
  bloom_h_input_image_idx: u32 = 0
  pick_dst_slot :: proc(forbidden_a, forbidden_b: int) -> int {
    for s in 0 ..< 3 {
      if s == forbidden_a do continue
      if s == forbidden_b do continue
      return s
    }
    return 0
  }
  for effect, i in self.effect_stack {
    is_last := i == len(self.effect_stack) - 1
    is_bloom_h := false
    is_bloom_v := false
    if e, ok := effect.(BloomEffect); ok {
      if e.direction < 0.5 do is_bloom_h = true
      else do is_bloom_v = true
    }
    dst_slot: int
    if !is_last {
      // V pass must not overwrite its composite source (bloom_h_input).
      forbidden_b := bloom_h_input_slot if is_bloom_v else -1
      dst_slot = pick_dst_slot(current_input_slot, forbidden_b)
    }
    dst_view := output_view
    if !is_last {
      dst_texture := gpu.get_texture_2d(
        texture_manager,
        self.images[dst_slot],
      )
      if dst_texture == nil {
        log.errorf(
          "Post-process image handle %v not found",
          self.images[dst_slot],
        )
        continue
      }
      gpu.image_discard_barrier(
        command_buffer,
        dst_texture.image,
        {.COLOR_ATTACHMENT_WRITE},
        {.COLOR_ATTACHMENT_OUTPUT},
        {.COLOR},
      )
      dst_view = dst_texture.view
    }
    if current_input_slot >= 0 {
      src_texture := gpu.get_texture_2d(
        texture_manager,
        self.images[current_input_slot],
      )
      if src_texture == nil {
        log.errorf(
          "Post-process source image handle %v not found",
          self.images[current_input_slot],
        )
        continue
      }
      gpu.memory_barrier(
        command_buffer,
        {.COLOR_ATTACHMENT_WRITE},
        {.SHADER_READ},
        {.COLOR_ATTACHMENT_OUTPUT},
        {.FRAGMENT_SHADER},
      )
    }
    input_image_index := current_input_image_idx
    gpu.begin_rendering(
      command_buffer,
      extent,
      nil,
      gpu.create_color_attachment_view(dst_view, .CLEAR, .STORE, BG_BLUE_GRAY),
    )
    effect_type := get_effect_type(effect)
    pipeline := self.pipelines_ldr[effect_type] if is_last else self.pipelines_hdr[effect_type]
    gpu.bind_graphics_pipeline(
      command_buffer,
      pipeline,
      self.pipeline_layouts[effect_type],
      texture_manager.descriptor_set,
    )
    base: BasePushConstant
    base.position_texture_index = position_texture_idx
    base.normal_texture_index = normal_texture_idx
    base.albedo_texture_index = albedo_texture_idx
    base.metallic_texture_index = metallic_texture_idx
    base.emissive_texture_index = emissive_texture_idx
    base.depth_texture_index = depth_texture_idx
    base.input_image_index = input_image_index
    // Create and push combined push constants based on effect type
    switch &e in effect {
    case BlurEffect:
      push_constant := BlurPushConstant {
        radius         = e.radius,
        direction      = e.direction,
        weight_falloff = e.weight_falloff,
      }
      push_constant.base = base
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(BlurPushConstant),
        &push_constant,
      )
    case GrayscaleEffect:
      push_constant := GrayscalePushConstant {
        weights  = e.weights,
        strength = e.strength,
      }
      push_constant.base = base
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(GrayscalePushConstant),
        &push_constant,
      )
    case ToneMapEffect:
      push_constant := ToneMapPushConstant {
        exposure = e.exposure,
        gamma    = e.gamma,
      }
      push_constant.base = base
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(ToneMapPushConstant),
        &push_constant,
      )
    case BloomEffect:
      push_constant := BloomPushConstant {
        threshold   = e.threshold,
        intensity   = e.intensity,
        blur_radius = e.blur_radius,
        direction   = e.direction,
      }
      push_constant.base = base
      // Bloom uses the padding slot as `original_image_index` — the scene as
      // it entered bloom, so V can composite blurred bright-pass against the
      // prior stack state (preserving outline, DOF, fog, ...). H is a no-op
      // for this field but we set it to its own input for consistency.
      push_constant.base.padding = input_image_index if is_bloom_h else bloom_h_input_image_idx
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(BloomPushConstant),
        &push_constant,
      )
    case OutlineEffect:
      push_constant := OutlinePushConstant {
        color     = e.color,
        thickness = e.thickness,
      }
      push_constant.base = base
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(OutlinePushConstant),
        &push_constant,
      )
    case FogEffect:
      push_constant: FogPushConstant
      push_constant.base = base
      push_constant.color = {e.color.x, e.color.y, e.color.z, 1.0} // Convert [3]f32 to [4]f32
      push_constant.density = e.density
      push_constant.start = e.start
      push_constant.end = e.end
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(FogPushConstant),
        &push_constant,
      )
    case CrossHatchEffect:
      push_constant := CrossHatchPushConstant {
        resolution       = e.resolution,
        hatch_offset_y   = e.hatch_offset_y,
        lum_threshold_01 = e.lum_threshold_01,
        lum_threshold_02 = e.lum_threshold_02,
        lum_threshold_03 = e.lum_threshold_03,
        lum_threshold_04 = e.lum_threshold_04,
      }
      push_constant.base = base
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(CrossHatchPushConstant),
        &push_constant,
      )
    case DoFEffect:
      push_constant := DoFPushConstant {
        focus_distance  = e.focus_distance,
        focus_range     = e.focus_range,
        blur_strength   = e.blur_strength,
        bokeh_intensity = e.bokeh_intensity,
      }
      push_constant.base = base
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(DoFPushConstant),
        &push_constant,
      )
    case:
      // No effect or nil effect - just copy input to output
      vk.CmdPushConstants(
        command_buffer,
        self.pipeline_layouts[effect_type],
        {.FRAGMENT},
        0,
        size_of(BasePushConstant),
        &base,
      )
    }
    vk.CmdDraw(command_buffer, 3, 1, 0, 0)
    vk.CmdEndRendering(command_buffer)
    if is_bloom_h {
      bloom_h_input_slot = current_input_slot
      bloom_h_input_image_idx = current_input_image_idx
    } else if is_bloom_v {
      bloom_h_input_slot = -1
      bloom_h_input_image_idx = 0
    }
    if !is_last {
      current_input_slot = dst_slot
      current_input_image_idx = self.images[dst_slot].index
    }
  }
}

