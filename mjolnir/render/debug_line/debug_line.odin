package debug_line

import "../../gpu"
import "core:log"
import "core:sync"
import vk "vendor:vulkan"

SHADER_VERT :: #load("../../shader/debug_line/vert.spv")
SHADER_FRAG :: #load("../../shader/debug_line/frag.spv")

FRAMES_IN_FLIGHT :: #config(FRAMES_IN_FLIGHT, 2)

MAX_VERTICES :: 131072
MAX_SEGMENTS :: 65536

Vertex :: struct {
  position: [3]f32,
  color:    [4]f32,
}

Segment :: struct {
  a, b:         [3]f32,
  color:        [4]f32,
  expiry:       f32,
  bypass_depth: bool,
}

PushConstant :: struct {
  camera_index: u32,
}

Renderer :: struct {
  pipeline_layout: vk.PipelineLayout,
  pipeline:        vk.Pipeline,
  vertex_buffers:  [FRAMES_IN_FLIGHT]gpu.MutableBuffer(Vertex),
  segments:        [dynamic]Segment,
  now:             f32,
  mu:              sync.Mutex,
}

init :: proc(
  self: ^Renderer,
  gctx: ^gpu.GPUContext,
  camera_set_layout: vk.DescriptorSetLayout,
) -> (
  ret: vk.Result,
) {
  for i in 0 ..< FRAMES_IN_FLIGHT {
    self.vertex_buffers[i] = gpu.create_mutable_buffer(
      gctx,
      Vertex,
      MAX_VERTICES,
      {.VERTEX_BUFFER},
    ) or_return
  }

  self.pipeline_layout = gpu.create_pipeline_layout(
    gctx,
    vk.PushConstantRange{stageFlags = {.VERTEX}, size = size_of(PushConstant)},
    camera_set_layout,
  ) or_return

  vert_module := gpu.create_shader_module(gctx.device, SHADER_VERT) or_return
  defer vk.DestroyShaderModule(gctx.device, vert_module, nil)
  frag_module := gpu.create_shader_module(gctx.device, SHADER_FRAG) or_return
  defer vk.DestroyShaderModule(gctx.device, frag_module, nil)

  binding := vk.VertexInputBindingDescription {
    binding   = 0,
    stride    = size_of(Vertex),
    inputRate = .VERTEX,
  }
  attrs := [?]vk.VertexInputAttributeDescription {
    {location = 0, binding = 0, format = .R32G32B32_SFLOAT, offset = u32(offset_of(Vertex, position))},
    {location = 1, binding = 0, format = .R32G32B32A32_SFLOAT, offset = u32(offset_of(Vertex, color))},
  }
  vinfo := vk.PipelineVertexInputStateCreateInfo {
    sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    vertexBindingDescriptionCount   = 1,
    pVertexBindingDescriptions      = &binding,
    vertexAttributeDescriptionCount = len(attrs),
    pVertexAttributeDescriptions    = raw_data(attrs[:]),
  }
  stages := gpu.create_vert_frag_stages(vert_module, frag_module)
  ia := vk.PipelineInputAssemblyStateCreateInfo {
    sType    = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    topology = .LINE_LIST,
  }
  // Depth test toggled dynamically per draw to fold overlay variant into the same pipeline.
  dyn_states := [?]vk.DynamicState{.VIEWPORT, .SCISSOR, .DEPTH_TEST_ENABLE}
  dyn_state := gpu.create_dynamic_state(dyn_states[:])

  pinfo := vk.GraphicsPipelineCreateInfo {
    sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
    stageCount          = len(stages),
    pStages             = raw_data(stages[:]),
    pVertexInputState   = &vinfo,
    pInputAssemblyState = &ia,
    pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
    pRasterizationState = &gpu.LINE_RASTERIZER,
    pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
    pDepthStencilState  = &gpu.READ_ONLY_DEPTH_STATE,
    pColorBlendState    = &gpu.COLOR_BLENDING_ADDITIVE,
    pDynamicState       = &dyn_state,
    layout              = self.pipeline_layout,
    pNext               = &gpu.STANDARD_RENDERING_INFO,
  }
  vk.CreateGraphicsPipelines(gctx.device, 0, 1, &pinfo, nil, &self.pipeline) or_return

  self.segments = make([dynamic]Segment, 0, 1024)
  log.debugf("debug_line renderer initialized")
  return .SUCCESS
}

shutdown :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) {
  delete(self.segments)
  for i in 0 ..< FRAMES_IN_FLIGHT {
    gpu.mutable_buffer_destroy(gctx.device, &self.vertex_buffers[i])
  }
  vk.DestroyPipeline(gctx.device, self.pipeline, nil)
  vk.DestroyPipelineLayout(gctx.device, self.pipeline_layout, nil)
  self.pipeline = 0
  self.pipeline_layout = 0
}

// Sentinel for one-frame draws; dropped after the first record() packs them.
ONE_FRAME_EXPIRY :: f32(-1.0)

set_time :: proc(self: ^Renderer, now: f32) {
  sync.mutex_lock(&self.mu)
  self.now = now
  sync.mutex_unlock(&self.mu)
}

resolve_expiry :: #force_inline proc(now, life: f32) -> f32 {
  return ONE_FRAME_EXPIRY if life <= 0 else now + life
}

add_segments :: proc(self: ^Renderer, segs: ..Segment) {
  if len(segs) == 0 do return
  sync.mutex_lock(&self.mu)
  defer sync.mutex_unlock(&self.mu)
  total := len(self.segments) + len(segs)
  if total > MAX_SEGMENTS {
    overflow := total - MAX_SEGMENTS
    if overflow >= len(self.segments) {
      clear(&self.segments)
    } else {
      copy(self.segments[:len(self.segments) - overflow], self.segments[overflow:])
      resize(&self.segments, len(self.segments) - overflow)
    }
  }
  append(&self.segments, ..segs)
}

record :: proc(
  self: ^Renderer,
  cmd: vk.CommandBuffer,
  frame_index: u32,
  camera_set: vk.DescriptorSet,
  camera_index: u32,
  color_view: vk.ImageView,
  extent: vk.Extent2D,
  depth_image: ^gpu.Image,
) {
  sync.mutex_lock(&self.mu)
  defer sync.mutex_unlock(&self.mu)

  if len(self.segments) == 0 do return

  vbuf := &self.vertex_buffers[frame_index]
  mapped := gpu.get_all(vbuf)
  now := self.now

  // Pass 1: pack depth-tested. Drop expired + packed-one-frame. Defer overlay to pass 2.
  vcount := 0
  write := 0
  for i in 0 ..< len(self.segments) {
    seg := self.segments[i]
    if seg.expiry > 0 && seg.expiry < now do continue
    if seg.bypass_depth {
      self.segments[write] = seg
      write += 1
      continue
    }
    if vcount + 2 > MAX_VERTICES {
      self.segments[write] = seg
      write += 1
      continue
    }
    mapped[vcount] = Vertex{position = seg.a, color = seg.color}
    mapped[vcount + 1] = Vertex{position = seg.b, color = seg.color}
    vcount += 2
    if seg.expiry != ONE_FRAME_EXPIRY {
      self.segments[write] = seg
      write += 1
    }
  }
  split := vcount

  // Pass 2: pack overlay from survivor prefix; compact in place.
  write2 := 0
  for i in 0 ..< write {
    seg := self.segments[i]
    if !seg.bypass_depth {
      self.segments[write2] = seg
      write2 += 1
      continue
    }
    if vcount + 2 > MAX_VERTICES {
      self.segments[write2] = seg
      write2 += 1
      continue
    }
    mapped[vcount] = Vertex{position = seg.a, color = seg.color}
    mapped[vcount + 1] = Vertex{position = seg.b, color = seg.color}
    vcount += 2
    if seg.expiry != ONE_FRAME_EXPIRY {
      self.segments[write2] = seg
      write2 += 1
    }
  }
  resize(&self.segments, write2)

  if vcount == 0 do return

  gpu.begin_rendering(
    cmd,
    extent,
    gpu.create_depth_attachment(
      depth_image,
      .LOAD,
      .STORE,
    ),
    gpu.create_color_attachment_view(color_view, .LOAD, .STORE),
  )
  gpu.set_viewport_scissor(cmd, extent)

  pc := PushConstant{camera_index = camera_index}
  offset := vk.DeviceSize(0)
  vbuf_h := vbuf.buffer
  vk.CmdBindVertexBuffers(cmd, 0, 1, &vbuf_h, &offset)
  descs := [?]vk.DescriptorSet{camera_set}
  vk.CmdBindDescriptorSets(cmd, .GRAPHICS, self.pipeline_layout, 0, 1, raw_data(descs[:]), 0, nil)
  vk.CmdPushConstants(cmd, self.pipeline_layout, {.VERTEX}, 0, size_of(PushConstant), &pc)

  vk.CmdBindPipeline(cmd, .GRAPHICS, self.pipeline)
  if split > 0 {
    vk.CmdSetDepthTestEnable(cmd, true)
    vk.CmdDraw(cmd, u32(split), 1, 0, 0)
  }
  if vcount > split {
    vk.CmdSetDepthTestEnable(cmd, false)
    vk.CmdDraw(cmd, u32(vcount - split), 1, u32(split), 0)
  }
  vk.CmdEndRendering(cmd)
}
