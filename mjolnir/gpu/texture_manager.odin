package gpu

import cont "../containers"
import "core:c"
import "core:log"
import "core:strings"
import stbi "vendor:stb/image"
import vk "vendor:vulkan"

MAX_TEXTURES :: 1000
MAX_CUBE_TEXTURES :: 200
Texture2DHandle :: distinct cont.Handle
TextureCubeHandle :: distinct cont.Handle

// TextureManager manages textures with handle pools.
// Handle.index maps directly to descriptor array indices.
TextureManager :: struct {
  images_2d:               cont.Pool(Image),
  images_cube:             cont.Pool(CubeImage),
  set_layout:              vk.DescriptorSetLayout,
  descriptor_set: vk.DescriptorSet,
}

// Initialize the texture manager: creates the descriptor set layout.
// Must be called once at startup, before pipeline layouts are built.
texture_manager_init :: proc(self: ^TextureManager, gctx: ^GPUContext) -> vk.Result {
  self.set_layout = create_descriptor_set_layout_array(
    gctx,
    {.SAMPLED_IMAGE, MAX_TEXTURES, {.FRAGMENT}},
    {.SAMPLER, MAX_SAMPLERS, {.FRAGMENT}},
    {.SAMPLED_IMAGE, MAX_CUBE_TEXTURES, {.FRAGMENT}},
  ) or_return
  return .SUCCESS
}

// Setup the texture manager: allocates the descriptor set and writes sampler descriptors.
// Samplers order: [nearest_clamp, linear_clamp, nearest_repeat, linear_repeat].
// Must be called each time the descriptor pool is (re)created.
texture_manager_setup :: proc(
  self: ^TextureManager,
  gctx: ^GPUContext,
  samplers: [MAX_SAMPLERS]vk.Sampler,
) -> vk.Result {
  // GPU images back descriptor slots the GPU samples for up to FRAMES_IN_FLIGHT
  // frames, so a freed slot/image must be retired, not reclaimed inline.
  cont.init(&self.images_2d, MAX_TEXTURES, FRAMES_IN_FLIGHT)
  cont.init(&self.images_cube, MAX_CUBE_TEXTURES, FRAMES_IN_FLIGHT)
  allocate_descriptor_set(gctx, &self.descriptor_set, &self.set_layout) or_return
  update_descriptor_set_array(
    gctx,
    self.descriptor_set,
    1,
    {.SAMPLER, vk.DescriptorImageInfo{sampler = samplers[0]}},
    {.SAMPLER, vk.DescriptorImageInfo{sampler = samplers[1]}},
    {.SAMPLER, vk.DescriptorImageInfo{sampler = samplers[2]}},
    {.SAMPLER, vk.DescriptorImageInfo{sampler = samplers[3]}},
  )
  return .SUCCESS
}

// Teardown the texture manager: destroys all GPU images and frees pool memory.
// The descriptor set handle is zeroed (bulk-freed by the caller via ResetDescriptorPool).
// Must be paired with texture_manager_setup.
@(private = "file")
retire_image_2d :: proc(img: ^Image, user: rawptr) {
  if img.image != 0 do image_destroy((^GPUContext)(user).device, img)
}

@(private = "file")
retire_image_cube :: proc(img: ^CubeImage, user: rawptr) {
  if img.image != 0 do cube_depth_texture_destroy((^GPUContext)(user).device, img)
}

// texture_manager_tick reclaims retired texture slots and destroys their vk
// images once no in-flight frame can still sample them. Call once per presented
// frame.
texture_manager_tick :: proc(self: ^TextureManager, gctx: ^GPUContext) {
  cont.pool_tick(&self.images_2d, gctx, retire_image_2d)
  cont.pool_tick(&self.images_cube, gctx, retire_image_cube)
}

texture_manager_teardown :: proc(self: ^TextureManager, gctx: ^GPUContext) {
  // Destroy any retired-but-not-yet-reclaimed images first; the caller has
  // already waited for the GPU to go idle, so grace no longer matters.
  cont.pool_flush(&self.images_2d, gctx, retire_image_2d)
  cont.pool_flush(&self.images_cube, gctx, retire_image_cube)
  for &entry in self.images_2d.entries do if entry.active {
    if entry.item.image != 0 do image_destroy(gctx.device, &entry.item)
  }
  for &entry in self.images_cube.entries do if entry.active {
    if entry.item.image != 0 do cube_depth_texture_destroy(gctx.device, &entry.item)
  }
  cont.destroy(self.images_2d, proc(_: ^Image) {})
  cont.destroy(self.images_cube, proc(_: ^CubeImage) {})
  self.images_2d = {}
  self.images_cube = {}
  self.descriptor_set = 0
}

// Shutdown the texture manager: destroys the descriptor set layout.
// Must be called once at shutdown, paired with texture_manager_init.
texture_manager_shutdown :: proc(self: ^TextureManager, gctx: ^GPUContext) {
  vk.DestroyDescriptorSetLayout(gctx.device, self.set_layout, nil)
  self.set_layout = 0
}

// Allocate a 2D texture with the given parameters
allocate_texture_2d :: proc(
  self: ^TextureManager,
  gctx: ^GPUContext,
  extent: vk.Extent2D,
  format: vk.Format,
  usage: vk.ImageUsageFlags,
  generate_mips: bool = false,
) -> (
  handle: Texture2DHandle,
  ret: vk.Result,
) {
  spec := image_spec_2d(extent, format, usage, generate_mips)
  img := image_create(gctx, spec) or_return
  gpu_handle, slot, ok := cont.alloc(&self.images_2d, Texture2DHandle)
  if !ok {
    image_destroy(gctx.device, &img)
    return {}, .ERROR_OUT_OF_DEVICE_MEMORY
  }
  slot^ = img
  // Update descriptor set
  set_texture_2d_descriptor(
    gctx,
    self.descriptor_set,
    gpu_handle.index,
    img.view,
  )
  return gpu_handle, .SUCCESS
}

// Allocate a 2D texture with initial data
allocate_texture_2d_with_data :: proc(
  tm: ^TextureManager,
  gctx: ^GPUContext,
  pixel_data: rawptr,
  data_size: vk.DeviceSize,
  extent: vk.Extent2D,
  format: vk.Format,
  usage: vk.ImageUsageFlags,
  generate_mips: bool = false,
) -> (
  handle: Texture2DHandle,
  ret: vk.Result,
) {
  spec := image_spec_2d(extent, format, usage, generate_mips)
  img: Image
  if generate_mips {
    img = image_create_with_mipmaps(
      gctx,
      spec,
      pixel_data,
      data_size,
    ) or_return
  } else {
    img = image_create_with_data(gctx, spec, pixel_data, data_size) or_return
  }
  gpu_handle, slot, ok := cont.alloc(&tm.images_2d, Texture2DHandle)
  if !ok {
    image_destroy(gctx.device, &img)
    return {}, .ERROR_OUT_OF_DEVICE_MEMORY
  }
  slot^ = img
  // Update descriptor set
  set_texture_2d_descriptor(
    gctx,
    tm.descriptor_set,
    gpu_handle.index,
    img.view,
  )
  return gpu_handle, .SUCCESS
}

// Free a 2D texture. The handle is invalidated immediately, but the slot and the
// underlying vk image are retired and destroyed by texture_manager_tick once no
// in-flight frame can still sample them.
free_texture_2d :: proc(tm: ^TextureManager, gctx: ^GPUContext, handle: $H) {
  gpu_handle := transmute(Texture2DHandle)handle
  cont.free_deferred(&tm.images_2d, gpu_handle)
}

// Get a pointer to a 2D texture
get_texture_2d :: proc(tm: ^TextureManager, handle: $H) -> ^Image {
  gpu_handle := transmute(Texture2DHandle)handle
  img, _ := cont.get(tm.images_2d, gpu_handle)
  return img
}

// Allocate a cube texture (typically used for depth cube maps)
allocate_texture_cube :: proc(
  tm: ^TextureManager,
  gctx: ^GPUContext,
  size: u32,
  format: vk.Format,
  usage: vk.ImageUsageFlags,
) -> (
  handle: TextureCubeHandle,
  ret: vk.Result,
) {
  h, img, ok := cont.alloc(&tm.images_cube, TextureCubeHandle)
  if !ok do return {}, .ERROR_OUT_OF_DEVICE_MEMORY
  // Initialize the cube image
  init_ret := cube_depth_texture_init(gctx, img, size, format, usage)
  if init_ret != .SUCCESS {
    cont.free(&tm.images_cube, h)
    return {}, init_ret
  }
  // Update descriptor set
  set_texture_cube_descriptor(
    gctx,
    tm.descriptor_set,
    h.index,
    img.view,
  )
  return h, .SUCCESS
}

// Allocate a color cube texture (for IBL precompute outputs).
// Optionally creates mip chain and STORAGE usage for compute writes.
allocate_texture_cube_color :: proc(
  tm: ^TextureManager,
  gctx: ^GPUContext,
  size: u32,
  format: vk.Format,
  mip_levels: u32 = 1,
  usage: vk.ImageUsageFlags = {.SAMPLED, .STORAGE, .TRANSFER_DST},
) -> (
  handle: TextureCubeHandle,
  ret: vk.Result,
) {
  h, img, ok := cont.alloc(&tm.images_cube, TextureCubeHandle)
  if !ok do return {}, .ERROR_OUT_OF_DEVICE_MEMORY
  init_ret := cube_color_texture_init(gctx, img, size, format, mip_levels, usage)
  if init_ret != .SUCCESS {
    cont.free(&tm.images_cube, h)
    return {}, init_ret
  }
  set_texture_cube_descriptor(gctx, tm.descriptor_set, h.index, img.view)
  return h, .SUCCESS
}

// Free a cube texture. Retired and destroyed by texture_manager_tick — see
// free_texture_2d.
free_texture_cube :: proc(tm: ^TextureManager, gctx: ^GPUContext, handle: $H) {
  gpu_handle := transmute(TextureCubeHandle)handle
  cont.free_deferred(&tm.images_cube, gpu_handle)
}

// Get a pointer to a cube texture
get_texture_cube :: proc(tm: ^TextureManager, handle: $H) -> ^CubeImage {
  gpu_handle := transmute(TextureCubeHandle)handle
  img, _ := cont.get(tm.images_cube, gpu_handle)
  return img
}

// Internal helper to update 2D texture descriptor
@(private)
set_texture_2d_descriptor :: proc(
  gctx: ^GPUContext,
  textures_descriptor_set: vk.DescriptorSet,
  index: u32,
  image_view: vk.ImageView,
) {
  if index >= MAX_TEXTURES do return
  if textures_descriptor_set == 0 do return
  update_descriptor_set_array_offset(
    gctx,
    textures_descriptor_set,
    0,
    index,
    {
      .SAMPLED_IMAGE,
      vk.DescriptorImageInfo {
        imageView = image_view,
        imageLayout = .GENERAL,
      },
    },
  )
}

// Internal helper to update cube texture descriptor
@(private)
set_texture_cube_descriptor :: proc(
  gctx: ^GPUContext,
  textures_descriptor_set: vk.DescriptorSet,
  index: u32,
  image_view: vk.ImageView,
) {
  if index >= MAX_CUBE_TEXTURES do return
  if textures_descriptor_set == 0 do return
  update_descriptor_set_array_offset(
    gctx,
    textures_descriptor_set,
    2,
    index,
    {
      .SAMPLED_IMAGE,
      vk.DescriptorImageInfo {
        imageView = image_view,
        imageLayout = .GENERAL,
      },
    },
  )
}

create_texture_2d_from_data :: proc(
  gctx: ^GPUContext,
  texture_manager: ^TextureManager,
  data: []u8,
  format: vk.Format = .R8G8B8A8_SRGB,
  generate_mips: bool = false,
) -> (
  handle: Texture2DHandle,
  ret: vk.Result,
) {
  width, height, channels: c.int
  pixels := stbi.load_from_memory(
    raw_data(data),
    c.int(len(data)),
    &width,
    &height,
    &channels,
    4,
  )
  if pixels == nil {
    log.errorf("Failed to decode texture data: %s", stbi.failure_reason())
    return {}, .ERROR_UNKNOWN
  }
  defer stbi.image_free(pixels)
  pixel_size := vk.DeviceSize(width * height * 4)
  return allocate_texture_2d_with_data(
    texture_manager,
    gctx,
    pixels,
    pixel_size,
    vk.Extent2D {u32(width), u32(height)},
    format,
    {.SAMPLED},
    generate_mips,
  )
}

create_texture_2d_from_path :: proc(
  gctx: ^GPUContext,
  texture_manager: ^TextureManager,
  path: string,
  format: vk.Format = .R8G8B8A8_SRGB,
  generate_mips: bool = false,
  usage: vk.ImageUsageFlags = {.SAMPLED},
  is_hdr: bool = false,
) -> (
  handle: Texture2DHandle,
  ret: vk.Result,
) {
  path_cstr := strings.clone_to_cstring(path)
  defer delete(path_cstr)
  width, height, channels: c.int
  pixel_data: rawptr
  data_size: vk.DeviceSize
  if is_hdr {
    pixels := stbi.loadf(path_cstr, &width, &height, &channels, 4)
    if pixels == nil {
      log.errorf(
        "Failed to load HDR texture '%s': %s",
        path,
        stbi.failure_reason(),
      )
      return {}, .ERROR_UNKNOWN
    }
    pixel_data = pixels
    data_size = vk.DeviceSize(width * height * 4 * size_of(f32))
  } else {
    pixels := stbi.load(path_cstr, &width, &height, &channels, 4)
    if pixels == nil {
      log.errorf(
        "Failed to load texture '%s': %s",
        path,
        stbi.failure_reason(),
      )
      return {}, .ERROR_UNKNOWN
    }
    pixel_data = pixels
    data_size = vk.DeviceSize(width * height * 4)
  }
  defer stbi.image_free(pixel_data)
  return allocate_texture_2d_with_data(
    texture_manager,
    gctx,
    pixel_data,
    data_size,
    vk.Extent2D {u32(width), u32(height)},
    format,
    usage,
    generate_mips,
  )
}
