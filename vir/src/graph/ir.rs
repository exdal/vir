use ash::vk;
use bitflags::bitflags;

use crate::{DomainFlag, Image, PassCallback};

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Access: u64 {
        const None = 0;
        const ColorRead = 1 << 0;
        const ColorWrite = 1 << 1;
        const ColorRW = Self::ColorRead.bits() | Self::ColorWrite.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

pub enum Constant {
    I32(i32),
    U32(u32),
    Extent2D(vk::Extent2D),
    Extent3D(vk::Extent3D),
}

pub enum IR {
    Constant(Constant),

    // Construct ops
    ConstructBuffer {
        result: ValueId,
        size: ValueId,
    },

    ConstructImage {
        result: ValueId,
        image: Image,
        image_view: vk::ImageView,
        extent: ValueId,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        base_mip: ValueId,
        mip_count: ValueId,
        base_layer: ValueId,
        layer_count: ValueId,
    },

    // Acquire ops
    AcquireSwapChain {
        result: ValueId,
        swapchain: vk::SwapchainKHR,
    },

    Acquire {
        result: ValueId,
        resource: ValueId,
        access: Access,
    },

    Release {
        resource: ValueId,
        src_domain: DomainFlag,
        dst_domain: DomainFlag,
    },

    // Pass ops
    CallOpaque {
        result: ValueId,
        args: Vec<ValueId>,
        returns: Vec<ValueId>,
        callback: PassCallback,
        domain: DomainFlag,
    },
}

// %attach = ImageAttachment %swap_attachment COLOR_RW
// %clear_image = BeginPass
//
//
// [0x0046] %reallocated_device_data = acquire<buffer>
// [0x0047] %device_data = acquire<buffer>
// [0x0048] %call_138, %call_139 = call $Graphics <realloc_copy_data> %device_data, %reallocated_device_data
// [0x0049] %splice_140, %splice_141 <- %call_138, %call_139
// [0x004a] %call_142 = call $Graphics <buffer_multi_swap> %splice_140
// [0x004b] %splice_143 <- %call_142
// [0x004c] %src = acquire<buffer>
// [0x004d] %src = acquire<buffer>
// [0x004e] %buffer_updates = construct<buffer[2]> %src, %src
// [0x004f] %splice_147 <- %buffer_updates
// [0x0050] %call_148, %call_149 = call $Graphics <buffer_multi_copy> %splice_147, %splice_143
// [0x0051] %splice_150, %splice_151 <- %call_148, %call_149
// [0x0052] %dst = construct<buffer> 64
// [0x0053] %splice_153 <- %dst
// [0x0054] %src = acquire<buffer>
// [0x0055] %call_155, %call_156 = call $Graphics <buffer_copy> %src, %splice_153
// [0x0056] %splice_157, %splice_158 <- %call_155, %call_156
// [0x0057] %updated_buffers = construct<buffer[2]> %splice_157, %splice_150
// [0x0058] %splice_160 <- %updated_buffers
// [0x0059] %construct_161 = construct<image> 800, 600, 1, <mem>, <mem>, 0, 1, 0, 1
// [0x005a] %construct_162 = construct<image> 800, 600, 1, <mem>, <mem>, 0, 1, 0, 1
// [0x005b] %construct_163 = construct<image> 800, 600, 1, <mem>, <mem>, 0, 1, 0, 1
// [0x005c] %construct_164 = construct<image[3]> %construct_163, %construct_162, %construct_161
// [0x005d] %construct_165 = construct<swapchain>
// [0x005e] %splice_166 = acquire<>
// [0x005f] %swp_img = acquire_next_image %splice_166
// [0x0060] %splice_168 <- %swp_img
// [0x0061] %call_169 = call $Graphics <clear image> %splice_168
// [0x0062] %splice_170 <- %call_169
// [0x0063] %call_171, %call_172 = call $Graphics <apply_buffer_updates> %splice_170, %splice_160
// [0x0064] %splice_173, %splice_174 <- %call_171, %call_172
// [0x0065] %call_175, %call_176 = call $Graphics <imgui> %splice_173, %splice_135
// [0x0066] %splice_177, %splice_178 <- %call_175, %call_176
// [0x0067] %splice_179 = release $Graphics -> $PE %splice_177
