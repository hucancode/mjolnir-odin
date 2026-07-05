package gpu

import alg "../algebra"
import "core:log"
import "core:math"
import vk "vendor:vulkan"

ImageType :: enum {
  D2,
  D2_ARRAY,
  D3,
  CUBE,
  CUBE_ARRAY,
}

ImageSpec :: struct {
  type:         ImageType,
  using extent:       vk.Extent2D,
  depth:        u32, // For 3D images
  array_layers: u32, // For array/cube images (cube = 6 layers)
  format:       vk.Format,
  mip_levels:   u32, // 0 = auto-calculate from dimensions
  tiling:       vk.ImageTiling,
  usage:        vk.ImageUsageFlags,
  memory_flags: vk.MemoryPropertyFlags,
  create_view:  bool,
  view_type:    vk.ImageViewType, // Auto-inferred if .UNDEFINED
  aspect_mask:  vk.ImageAspectFlags, // Auto-inferred from format if empty
}

Image :: struct {
  image:  vk.Image,
  memory: vk.DeviceMemory,
  spec:   ImageSpec,
  view:   vk.ImageView,
}

infer_view_type :: proc(
  img_type: ImageType,
  array_layers: u32,
) -> vk.ImageViewType {
  switch img_type {
  case .D2:
    return array_layers > 1 ? .D2_ARRAY : .D2
  case .D2_ARRAY:
    return .D2_ARRAY
  case .D3:
    return .D3
  case .CUBE:
    return .CUBE
  case .CUBE_ARRAY:
    return .CUBE_ARRAY
  }
  return .D2
}

infer_image_type :: proc(view_type: vk.ImageViewType) -> vk.ImageType {
  switch view_type {
  case .D1, .D1_ARRAY:
    return .D1
  case .D2, .D2_ARRAY:
    return .D2
  case .D3:
    return .D3
  case .CUBE, .CUBE_ARRAY:
    return .D2
  }
  return .D2
}

infer_aspect_mask :: proc(format: vk.Format) -> vk.ImageAspectFlags {
  #partial switch format {
  case .D16_UNORM, .D32_SFLOAT, .X8_D24_UNORM_PACK32:
    return {.DEPTH}
  case .D16_UNORM_S8_UINT, .D24_UNORM_S8_UINT, .D32_SFLOAT_S8_UINT:
    return {.DEPTH, .STENCIL}
  case .S8_UINT:
    return {.STENCIL}
  case:
    return {.COLOR}
  }
}

// Calculate optimal mip levels for given dimensions
calculate_mip_levels :: proc(width, height: u32) -> u32 {
  return alg.log2_greater_than(max(width, height))
}

validate_spec :: proc(spec: ^ImageSpec) {
  // Auto-calculate mip levels if not specified
  if spec.mip_levels == 0 {
    spec.mip_levels = 1
  }
  // Infer view type if not specified (0 is the zero value)
  if spec.view_type == {} {
    spec.view_type = infer_view_type(spec.type, spec.array_layers)
  }
  // Infer aspect mask if not specified
  if card(spec.aspect_mask) == 0 {
    spec.aspect_mask = infer_aspect_mask(spec.format)
  }
  // Set default array layers for cube maps
  if spec.type == .CUBE && spec.array_layers == 0 {
    spec.array_layers = 6
  } else if spec.type == .CUBE_ARRAY && spec.array_layers < 6 {
    spec.array_layers = 6
  }
  // Set default depth for 2D images
  if (spec.type == .D2 ||
       spec.type == .D2_ARRAY ||
       spec.type == .CUBE ||
       spec.type == .CUBE_ARRAY) &&
     spec.depth == 0 {
    spec.depth = 1
  }
  // Set default create_view to true if not explicitly disabled
  if !spec.create_view && spec.view_type != {} {
    spec.create_view = true
  }
}

image_create :: proc(
  gctx: ^GPUContext,
  spec: ImageSpec,
) -> (
  img: Image,
  ret: vk.Result,
) {
  img.spec = spec
  validate_spec(&img.spec)
  // Handle CUBE flag for cube maps
  flags: vk.ImageCreateFlags
  if img.spec.type == .CUBE || img.spec.type == .CUBE_ARRAY {
    flags = {.CUBE_COMPATIBLE}
  }
  create_info := vk.ImageCreateInfo {
    sType         = .IMAGE_CREATE_INFO,
    flags         = flags,
    imageType     = infer_image_type(img.spec.view_type),
    extent        = {img.spec.extent.width, img.spec.extent.height, img.spec.depth},
    mipLevels     = img.spec.mip_levels,
    arrayLayers   = max(img.spec.array_layers, 1),
    format        = img.spec.format,
    tiling        = img.spec.tiling,
    initialLayout = .UNDEFINED,
    usage         = img.spec.usage,
    sharingMode   = .EXCLUSIVE,
    samples       = {._1},
  }
  vk.CreateImage(gctx.device, &create_info, nil, &img.image) or_return
  mem_reqs: vk.MemoryRequirements
  vk.GetImageMemoryRequirements(gctx.device, img.image, &mem_reqs)
  img.memory = allocate_memory(gctx, mem_reqs, img.spec.memory_flags) or_return
  vk.BindImageMemory(gctx.device, img.image, img.memory, 0) or_return
  // Create default view if requested
  if img.spec.create_view {
    view_info := vk.ImageViewCreateInfo {
      sType = .IMAGE_VIEW_CREATE_INFO,
      image = img.image,
      viewType = img.spec.view_type,
      format = img.spec.format,
      components = {
        r = .IDENTITY,
        g = .IDENTITY,
        b = .IDENTITY,
        a = .IDENTITY,
      },
      subresourceRange = {
        aspectMask = img.spec.aspect_mask,
        levelCount = img.spec.mip_levels,
        layerCount = max(img.spec.array_layers, 1),
      },
    }
    vk.CreateImageView(gctx.device, &view_info, nil, &img.view) or_return
  }
  log.debugf(
    "Created image %v %dx%d (mips=%d, layers=%d) 0x%x",
    img.spec.format,
    img.spec.extent.width,
    img.spec.extent.height,
    img.spec.mip_levels,
    img.spec.array_layers,
    img.image,
  )
  return img, .SUCCESS
}

image_create_with_data :: proc(
  gctx: ^GPUContext,
  spec: ImageSpec,
  data: rawptr,
  size: vk.DeviceSize,
) -> (
  img: Image,
  ret: vk.Result,
) {
  // Ensure TRANSFER_DST is in usage flags
  modified_spec := spec
  modified_spec.usage |= {.TRANSFER_DST}
  img = image_create(gctx, modified_spec) or_return
  // Create staging buffer
  staging := create_mutable_buffer(
    gctx,
    u8,
    int(size),
    {.TRANSFER_SRC},
    data,
  ) or_return
  defer mutable_buffer_destroy(gctx.device, &staging)
  cmd_buffer := begin_single_time_command(gctx) or_return
  image_discard_barrier(
    cmd_buffer,
    img.image,
    {.TRANSFER_WRITE},
    {.TRANSFER},
    img.spec.aspect_mask,
    layer_count = max(img.spec.array_layers, 1),
  )
  // Copy buffer to image
  region := vk.BufferImageCopy {
    imageSubresource = {
      aspectMask = img.spec.aspect_mask,
      mipLevel = 0,
      layerCount = max(img.spec.array_layers, 1),
    },
    imageExtent = {img.spec.extent.width, img.spec.extent.height, img.spec.depth},
  }
  vk.CmdCopyBufferToImage(
    cmd_buffer,
    staging.buffer,
    img.image,
    .GENERAL,
    1,
    &region,
  )
  // Make transfer writes visible to shader sampling
  memory_barrier(
    cmd_buffer,
    {.TRANSFER_WRITE},
    {.SHADER_READ},
    {.TRANSFER},
    {.FRAGMENT_SHADER},
  )
  end_single_time_command(gctx, &cmd_buffer) or_return
  return img, .SUCCESS
}

image_create_with_mipmaps :: proc(
  gctx: ^GPUContext,
  spec: ImageSpec,
  data: rawptr,
  size: vk.DeviceSize,
) -> (
  img: Image,
  ret: vk.Result,
) {
  // Auto-calculate mip levels
  modified_spec := spec
  if modified_spec.mip_levels == 0 {
    modified_spec.mip_levels = calculate_mip_levels(spec.extent.width, spec.extent.height)
  }
  // Ensure required usage flags for mipmap generation
  modified_spec.usage |= {.TRANSFER_DST, .TRANSFER_SRC}
  // Verify format supports linear filtering for blit
  format_props: vk.FormatProperties
  vk.GetPhysicalDeviceFormatProperties(
    gctx.physical_device,
    modified_spec.format,
    &format_props,
  )
  if .SAMPLED_IMAGE_FILTER_LINEAR not_in format_props.optimalTilingFeatures {
    log.errorf(
      "Format %v does not support linear blitting for mipmaps",
      modified_spec.format,
    )
    return img, .ERROR_FORMAT_NOT_SUPPORTED
  }
  img = image_create(gctx, modified_spec) or_return
  // Create staging buffer
  staging := create_mutable_buffer(
    gctx,
    u8,
    int(size),
    {.TRANSFER_SRC},
    data,
  ) or_return
  defer mutable_buffer_destroy(gctx.device, &staging)
  cmd_buffer := begin_single_time_command(gctx) or_return
  image_discard_barrier(
    cmd_buffer,
    img.image,
    {.TRANSFER_WRITE},
    {.TRANSFER},
    img.spec.aspect_mask,
    level_count = img.spec.mip_levels,
    layer_count = max(img.spec.array_layers, 1),
  )
  // Copy base mip level from staging buffer
  region := vk.BufferImageCopy {
    imageSubresource = {
      aspectMask = img.spec.aspect_mask,
      layerCount = max(img.spec.array_layers, 1),
    },
    imageExtent = {img.spec.extent.width, img.spec.extent.height, img.spec.depth},
  }
  vk.CmdCopyBufferToImage(
    cmd_buffer,
    staging.buffer,
    img.image,
    .GENERAL,
    1,
    &region,
  )
  // Generate mipmaps
  mip_width := i32(img.spec.width)
  mip_height := i32(img.spec.height)
  for i in 1 ..< img.spec.mip_levels {
    // Previous mip's writes must land before this blit reads it
    memory_barrier(
      cmd_buffer,
      {.TRANSFER_WRITE},
      {.TRANSFER_READ},
      {.TRANSFER},
      {.TRANSFER},
    )
    // Blit from previous mip to current mip
    blit := vk.ImageBlit {
      srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
      srcSubresource = {
        aspectMask = img.spec.aspect_mask,
        mipLevel = i - 1,
        layerCount = max(img.spec.array_layers, 1),
      },
      dstOffsets = {
        {0, 0, 0},
        {max(mip_width / 2, 1), max(mip_height / 2, 1), 1},
      },
      dstSubresource = {
        aspectMask = img.spec.aspect_mask,
        mipLevel = i,
        layerCount = max(img.spec.array_layers, 1),
      },
    }
    vk.CmdBlitImage(
      cmd_buffer,
      img.image,
      .GENERAL,
      img.image,
      .GENERAL,
      1,
      &blit,
      .LINEAR,
    )
    mip_width = max(mip_width / 2, 1)
    mip_height = max(mip_height / 2, 1)
  }
  // Make all mip writes visible to shader sampling
  memory_barrier(
    cmd_buffer,
    {.TRANSFER_WRITE},
    {.SHADER_READ},
    {.TRANSFER},
    {.FRAGMENT_SHADER},
  )
  end_single_time_command(gctx, &cmd_buffer) or_return
  log.debugf(
    "Generated %d mip levels for image 0x%x",
    img.spec.mip_levels,
    img.image,
  )
  return img, .SUCCESS
}

image_create_view :: proc(
  device: vk.Device,
  img: ^Image,
  view_type: vk.ImageViewType,
  base_mip: u32 = 0,
  mip_count: u32 = 1,
  base_layer: u32 = 0,
  layer_count: u32 = 1,
) -> (
  view: vk.ImageView,
  ret: vk.Result,
) {
  view_info := vk.ImageViewCreateInfo {
    sType = .IMAGE_VIEW_CREATE_INFO,
    image = img.image,
    viewType = view_type,
    format = img.spec.format,
    components = {r = .IDENTITY, g = .IDENTITY, b = .IDENTITY, a = .IDENTITY},
    subresourceRange = {
      aspectMask = img.spec.aspect_mask,
      baseMipLevel = base_mip,
      levelCount = mip_count,
      baseArrayLayer = base_layer,
      layerCount = layer_count,
    },
  }
  ret = vk.CreateImageView(device, &view_info, nil, &view)
  return
}

image_destroy :: proc(device: vk.Device, img: ^Image) {
  vk.DestroyImageView(device, img.view, nil)
  img.view = 0
  vk.DestroyImage(device, img.image, nil)
  img.image = 0
  vk.FreeMemory(device, img.memory, nil)
  img.memory = 0
}

image_spec_2d :: proc(
  extent: vk.Extent2D,
  format: vk.Format,
  usage: vk.ImageUsageFlags,
  mipmaps := false,
) -> ImageSpec {
  mip_levels := u32(1)
  if mipmaps {
    mip_levels = calculate_mip_levels(extent.width, extent.height)
  }
  return ImageSpec {
    type = .D2,
    extent = extent,
    depth = 1,
    array_layers = 1,
    format = format,
    mip_levels = mip_levels,
    tiling = .OPTIMAL,
    usage = usage,
    memory_flags = {.DEVICE_LOCAL},
    create_view = true,
  }
}

image_spec_cube :: proc(
  size: u32,
  format: vk.Format,
  usage: vk.ImageUsageFlags,
  mipmaps := false,
) -> ImageSpec {
  mip_levels := u32(1)
  if mipmaps {
    mip_levels = calculate_mip_levels(size, size)
  }
  return ImageSpec {
    type = .CUBE,
    width = size,
    height = size,
    depth = 1,
    array_layers = 6,
    format = format,
    mip_levels = mip_levels,
    tiling = .OPTIMAL,
    usage = usage,
    memory_flags = {.DEVICE_LOCAL},
    create_view = true,
  }
}
