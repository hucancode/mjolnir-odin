package particles_render

import "../../gpu"
import particles_compute "../particles_compute"
import "../shared"
import "core:log"
import vk "vendor:vulkan"

Particle :: particles_compute.Particle

SHADER_PARTICLE_VERT := #load("../../shader/particle/vert.spv")
SHADER_PARTICLE_FRAG := #load("../../shader/particle/frag.spv")
TEXTURE_DEFAULT :: #load("../../assets/circle_03.png")

Renderer :: struct {
	render_pipeline_layout: vk.PipelineLayout,
	render_pipeline:        vk.Pipeline,
	default_texture_index:  u32,
}

init :: proc(
	self: ^Renderer,
	gctx: ^gpu.GPUContext,
	texture_manager: ^gpu.TextureManager,
	camera_set_layout: vk.DescriptorSetLayout,
	textures_set_layout: vk.DescriptorSetLayout,
) -> (
	ret: vk.Result,
) {
	log.debugf("Initializing particle render renderer")
	default_texture_handle := gpu.create_texture_2d_from_data(
		gctx,
		texture_manager,
		TEXTURE_DEFAULT,
	) or_return
	defer if ret != .SUCCESS {
		gpu.free_texture_2d(texture_manager, gctx, default_texture_handle)
	}
	self.default_texture_index = default_texture_handle.index

	create_render_pipeline(gctx, self, camera_set_layout, textures_set_layout) or_return
	defer if ret != .SUCCESS {
		vk.DestroyPipelineLayout(gctx.device, self.render_pipeline_layout, nil)
		vk.DestroyPipeline(gctx.device, self.render_pipeline, nil)
	}

	return .SUCCESS
}

shutdown :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) {
	vk.DestroyPipeline(gctx.device, self.render_pipeline, nil)
	vk.DestroyPipelineLayout(gctx.device, self.render_pipeline_layout, nil)
}

create_render_pipeline :: proc(
	gctx: ^gpu.GPUContext,
	self: ^Renderer,
	camera_set_layout: vk.DescriptorSetLayout,
	textures_set_layout: vk.DescriptorSetLayout,
) -> (
	ret: vk.Result,
) {
	self.render_pipeline_layout = gpu.create_pipeline_layout(
		gctx,
		vk.PushConstantRange{stageFlags = {.VERTEX}, size = size_of(u32)},
		camera_set_layout,
		textures_set_layout,
	) or_return
	defer if ret != .SUCCESS {
		vk.DestroyPipelineLayout(gctx.device, self.render_pipeline_layout, nil)
	}

	vertex_binding := vk.VertexInputBindingDescription {
		binding   = 0,
		stride    = size_of(Particle),
		inputRate = .VERTEX,
	}

	vertex_attributes := [?]vk.VertexInputAttributeDescription {
		{
			location = 0,
			binding = 0,
			format = .R32G32B32A32_SFLOAT,
			offset = u32(offset_of(Particle, position)),
		},
		{
			location = 1,
			binding = 0,
			format = .R32G32B32A32_SFLOAT,
			offset = u32(offset_of(Particle, color)),
		},
		{
			location = 2,
			binding = 0,
			format = .R32G32B32A32_SFLOAT,
			offset = u32(offset_of(Particle, size)),
		},
		{
			location = 3,
			binding = 0,
			format = .R32_UINT,
			offset = u32(offset_of(Particle, texture_index)),
		},
	}

	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 1,
		pVertexBindingDescriptions      = &vertex_binding,
		vertexAttributeDescriptionCount = len(vertex_attributes),
		pVertexAttributeDescriptions    = raw_data(vertex_attributes[:]),
	}

	vert_module := gpu.create_shader_module(gctx.device, SHADER_PARTICLE_VERT) or_return
	frag_module := gpu.create_shader_module(gctx.device, SHADER_PARTICLE_FRAG) or_return
	defer vk.DestroyShaderModule(gctx.device, vert_module, nil)
	defer vk.DestroyShaderModule(gctx.device, frag_module, nil)

	shader_stages := gpu.create_vert_frag_stages(
		vert_module,
		frag_module,
		&shared.SHADER_SPEC_CONSTANTS,
	)

	pipeline_info := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount          = len(shader_stages),
		pStages             = raw_data(shader_stages[:]),
		pVertexInputState   = &vertex_input_info,
		pInputAssemblyState = &gpu.POINT_INPUT_ASSEMBLY,
		pViewportState      = &gpu.STANDARD_VIEWPORT_STATE,
		pRasterizationState = &gpu.DOUBLE_SIDED_RASTERIZER,
		pMultisampleState   = &gpu.STANDARD_MULTISAMPLING,
		pColorBlendState    = &gpu.COLOR_BLENDING_ADDITIVE,
		pDynamicState       = &gpu.STANDARD_DYNAMIC_STATES,
		pDepthStencilState  = &gpu.READ_ONLY_DEPTH_STATE,
		layout              = self.render_pipeline_layout,
		pNext               = &gpu.HDR_STANDARD_RENDERING_INFO,
	}

	vk.CreateGraphicsPipelines(
		gctx.device,
		0,
		1,
		&pipeline_info,
		nil,
		&self.render_pipeline,
	) or_return

	return .SUCCESS
}

setup :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) -> vk.Result {
	return .SUCCESS
}

teardown :: proc(self: ^Renderer, gctx: ^gpu.GPUContext) {
}

// record draws particles into the camera's color/depth attachments. Owns its
// own begin_rendering / end_rendering scope.
record :: proc(
	self: ^Renderer,
	command_buffer: vk.CommandBuffer,
	camera_index: u32,
	color_handle: gpu.Texture2DHandle,
	depth_handle: gpu.Texture2DHandle,
	texture_manager: ^gpu.TextureManager,
	cameras_descriptor_set: vk.DescriptorSet,
	textures_descriptor_set: vk.DescriptorSet,
	compact_particle_buffer: vk.Buffer,
	draw_command_buffer: vk.Buffer,
) {
	color_texture := gpu.get_texture_2d(texture_manager, color_handle)
	if color_texture == nil {
		log.error("Particle renderer missing color attachment")
		return
	}
	depth_texture := gpu.get_texture_2d(texture_manager, depth_handle)
	if depth_texture == nil {
		log.error("Particle renderer missing depth attachment")
		return
	}
	gpu.begin_rendering(
		command_buffer,
		depth_texture.spec.extent,
		gpu.create_depth_attachment(
			depth_texture,
			.LOAD,
			.STORE,
		),
		gpu.create_color_attachment(color_texture, .LOAD, .STORE),
	)
	gpu.set_viewport_scissor(command_buffer, depth_texture.spec.extent)
	gpu.bind_graphics_pipeline(
		command_buffer,
		self.render_pipeline,
		self.render_pipeline_layout,
		cameras_descriptor_set,
		textures_descriptor_set,
	)
	camera_idx := camera_index
	vk.CmdPushConstants(
		command_buffer,
		self.render_pipeline_layout,
		{.VERTEX},
		0,
		size_of(u32),
		&camera_idx,
	)
	offset: vk.DeviceSize = 0
	buffer := compact_particle_buffer
	vk.CmdBindVertexBuffers(command_buffer, 0, 1, &buffer, &offset)
	vk.CmdDrawIndirect(
		command_buffer,
		draw_command_buffer,
		0,
		1,
		size_of(vk.DrawIndirectCommand),
	)
	vk.CmdEndRendering(command_buffer)
}
