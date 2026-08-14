use ash::vk;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Access: u64 {
        const None = 1 << 0;
        const ColorRead = 1 << 1;
        const ColorWrite = 1 << 2;
        const ColorRW = Self::ColorRead.bits() | Self::ColorWrite.bits();
        const DepthStencilRead = 1 << 3;
        const DepthStencilWrite = 1 << 4;
        const DepthStencilRW = Self::DepthStencilRead.bits() | Self::DepthStencilWrite.bits();
        const VertexSampled = 1 << 5;
        const VertexRead = 1 << 6;
        const AttributeRead = 1 << 7;
        const IndexRead = 1 << 8;
        const IndirectRead = 1 << 9;
        const VertexUniformRead = 1 << 10;
        const FragmentSampled = 1 << 11;
        const FragmentRead = 1 << 12;
        const FragmentWrite = 1 << 13;
        const FragmentRW = Self::FragmentRead.bits() | Self::FragmentWrite.bits();
        const FragmentUniformRead = 1 << 14;
        const CopyRead = 1 << 15;
        const CopyWrite = 1 << 16;
        const CopyRW = Self::CopyRead.bits() | Self::CopyWrite.bits();
        const BlitRead = 1 << 17;
        const BlitWrite = 1 << 18;
        const BlitRW = Self::BlitRead.bits() | Self::BlitWrite.bits();
        const Clear = 1 << 20;
        const ResolveRead = 1 << 21;
        const ResolveWrite = 1 << 22;
        const ResolveRW = Self::ResolveRead.bits() | Self::ResolveWrite.bits();
        const TransferRead = Self::CopyRead.bits() | Self::BlitRead.bits() | Self::ResolveRead.bits();
        const TransferWrite = Self::CopyWrite.bits() | Self::BlitWrite.bits() | Self::Clear.bits() | Self::ResolveWrite.bits();
        const TransferRW = Self::TransferRead.bits() | Self::TransferWrite.bits();
        const ComputeRead = 1 << 23;
        const ComputeWrite = 1 << 24;
        const ComputeRW = Self::ComputeRead.bits() | Self::ComputeWrite.bits();
        const ComputeSampled = 1 << 25;
        const ComputeUniformRead = 1 << 26;
        const RayTracingRead = 1 << 27;
        const RayTracingWrite = 1 << 28;
        const RayTracingRW = Self::RayTracingRead.bits() | Self::RayTracingWrite.bits();
        const RayTracingSampled = 1 << 29;
        const RayTracingUniformRead = 1 << 30;
        const AccelerationStructureBuildRead = 1 << 31;
        const AccelerationStructureBuildWrite = 1 << 32;
        const AccelerationStructureBuildRW = Self::AccelerationStructureBuildRead.bits() | Self::AccelerationStructureBuildWrite.bits();
        const HostRead = 1 << 33;
        const HostWrite = 1 << 34;
        const HostRW = Self::HostRead.bits() | Self::HostWrite.bits();
        const MemoryRead = 1 << 35;
        const MemoryWrite = 1 << 36;
        const MemoryRW = Self::MemoryRead.bits() | Self::MemoryWrite.bits();
        const Present = 1 << 37;
        const TessellationSampled = 1 << 38;
        const TessellationRead = 1 << 39;
        const TessellationUniformRead = 1 << 40;
        const Writes = Self::ColorWrite.bits()
            | Self::DepthStencilWrite.bits()
            | Self::FragmentWrite.bits()
            | Self::ComputeWrite.bits()
            | Self::RayTracingWrite.bits()
            | Self::CopyWrite.bits()
            | Self::BlitWrite.bits()
            | Self::Clear.bits()
            | Self::ResolveWrite.bits()
            | Self::AccelerationStructureBuildWrite.bits()
            | Self::HostWrite.bits()
            | Self::MemoryWrite.bits();
    }
}

impl Access {
    pub fn writes(self) -> bool { self.intersects(Access::Writes) }

    pub fn reads(self) -> bool { !self.difference(Access::Writes | Access::None).is_empty() }
}

impl From<Access> for vk::AccessFlags2 {
    fn from(access: Access) -> Self {
        let mut flags = vk::AccessFlags2::empty();
        if access.contains(Access::ColorRead) {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        }
        if access.contains(Access::ColorWrite) {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }
        if access.contains(Access::DepthStencilRead) {
            flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        if access.contains(Access::DepthStencilWrite) {
            flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        if access.contains(Access::VertexSampled) {
            flags |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if access.contains(Access::VertexRead) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if access.contains(Access::AttributeRead) {
            flags |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
        }
        if access.contains(Access::IndexRead) {
            flags |= vk::AccessFlags2::INDEX_READ;
        }
        if access.contains(Access::IndirectRead) {
            flags |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
        }
        if access.contains(Access::VertexUniformRead) {
            flags |= vk::AccessFlags2::UNIFORM_READ;
        }
        if access.contains(Access::FragmentSampled) {
            flags |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if access.contains(Access::FragmentRead) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if access.contains(Access::FragmentWrite) {
            flags |= vk::AccessFlags2::SHADER_WRITE;
        }
        if access.contains(Access::FragmentUniformRead) {
            flags |= vk::AccessFlags2::UNIFORM_READ;
        }
        if access.contains(Access::CopyRead) {
            flags |= vk::AccessFlags2::TRANSFER_READ;
        }
        if access.contains(Access::CopyWrite) {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if access.contains(Access::BlitRead) {
            flags |= vk::AccessFlags2::TRANSFER_READ;
        }
        if access.contains(Access::BlitWrite) {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if access.contains(Access::Clear) {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if access.contains(Access::ResolveRead) {
            flags |= vk::AccessFlags2::TRANSFER_READ;
        }
        if access.contains(Access::ResolveWrite) {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if access.contains(Access::ComputeRead) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if access.contains(Access::ComputeWrite) {
            flags |= vk::AccessFlags2::SHADER_WRITE;
        }
        if access.contains(Access::ComputeSampled) {
            flags |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if access.contains(Access::ComputeUniformRead) {
            flags |= vk::AccessFlags2::UNIFORM_READ;
        }
        if access.contains(Access::RayTracingRead) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if access.contains(Access::RayTracingWrite) {
            flags |= vk::AccessFlags2::SHADER_WRITE;
        }
        if access.contains(Access::RayTracingSampled) {
            flags |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if access.contains(Access::RayTracingUniformRead) {
            flags |= vk::AccessFlags2::UNIFORM_READ;
        }
        if access.contains(Access::AccelerationStructureBuildRead) {
            flags |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
        }
        if access.contains(Access::AccelerationStructureBuildWrite) {
            flags |= vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
        }
        if access.contains(Access::HostRead) {
            flags |= vk::AccessFlags2::HOST_READ;
        }
        if access.contains(Access::HostWrite) {
            flags |= vk::AccessFlags2::HOST_WRITE;
        }
        if access.contains(Access::MemoryRead) {
            flags |= vk::AccessFlags2::MEMORY_READ;
        }
        if access.contains(Access::MemoryWrite) {
            flags |= vk::AccessFlags2::MEMORY_WRITE;
        }
        if access.contains(Access::TessellationSampled) {
            flags |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if access.contains(Access::TessellationRead) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if access.contains(Access::TessellationUniformRead) {
            flags |= vk::AccessFlags2::UNIFORM_READ;
        }
        flags
    }
}

impl From<Access> for vk::PipelineStageFlags2 {
    fn from(access: Access) -> Self {
        let mut flags = vk::PipelineStageFlags2::empty();
        if access.contains(Access::ColorRead) {
            flags |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        }
        if access.contains(Access::ColorWrite) {
            flags |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        }
        if access.contains(Access::DepthStencilRead) {
            flags |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
        }
        if access.contains(Access::DepthStencilWrite) {
            flags |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        }
        if access.contains(Access::VertexSampled) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if access.contains(Access::VertexRead) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if access.contains(Access::VertexUniformRead) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if access.contains(Access::AttributeRead) {
            flags |= vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT;
        }
        if access.contains(Access::IndexRead) {
            flags |= vk::PipelineStageFlags2::INDEX_INPUT;
        }
        if access.contains(Access::IndirectRead) {
            flags |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        }
        if access.contains(Access::FragmentSampled) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if access.contains(Access::FragmentRead) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if access.contains(Access::FragmentWrite) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if access.contains(Access::FragmentUniformRead) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if access.contains(Access::CopyRead) {
            flags |= vk::PipelineStageFlags2::COPY;
        }
        if access.contains(Access::CopyWrite) {
            flags |= vk::PipelineStageFlags2::COPY;
        }
        if access.contains(Access::BlitRead) {
            flags |= vk::PipelineStageFlags2::BLIT;
        }
        if access.contains(Access::BlitWrite) {
            flags |= vk::PipelineStageFlags2::BLIT;
        }
        if access.contains(Access::Clear) {
            flags |= vk::PipelineStageFlags2::CLEAR;
        }
        if access.contains(Access::ResolveRead) {
            flags |= vk::PipelineStageFlags2::RESOLVE;
        }
        if access.contains(Access::ResolveWrite) {
            flags |= vk::PipelineStageFlags2::RESOLVE;
        }
        if access.contains(Access::ComputeRead) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if access.contains(Access::ComputeWrite) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if access.contains(Access::ComputeSampled) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if access.contains(Access::ComputeUniformRead) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if access.contains(Access::RayTracingRead) {
            flags |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        if access.contains(Access::RayTracingWrite) {
            flags |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        if access.contains(Access::RayTracingSampled) {
            flags |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        if access.contains(Access::RayTracingUniformRead) {
            flags |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        }
        if access.contains(Access::AccelerationStructureBuildRead) {
            flags |= vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
        }
        if access.contains(Access::AccelerationStructureBuildWrite) {
            flags |= vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
        }
        if access.contains(Access::HostRead) {
            flags |= vk::PipelineStageFlags2::HOST;
        }
        if access.contains(Access::HostWrite) {
            flags |= vk::PipelineStageFlags2::HOST;
        }
        if access.contains(Access::MemoryRead) {
            flags |= vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        if access.contains(Access::MemoryWrite) {
            flags |= vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        if access.contains(Access::Present) {
            flags |= vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        if access.contains(Access::TessellationSampled) {
            flags |= vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER;
        }
        if access.contains(Access::TessellationRead) {
            flags |= vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER;
        }
        if access.contains(Access::TessellationUniformRead) {
            flags |= vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER
                | vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER;
        }
        flags
    }
}

impl From<Access> for vk::ImageLayout {
    fn from(access: Access) -> Self {
        if access.contains(Access::Present) {
            return vk::ImageLayout::PRESENT_SRC_KHR;
        }
        if access.contains(Access::ColorRW) || access.contains(Access::ColorWrite) || access.contains(Access::ColorRead)
        {
            return vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        }
        if access.contains(Access::DepthStencilRW) || access.contains(Access::DepthStencilWrite) {
            return vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
        if access.contains(Access::DepthStencilRead) {
            return vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        }
        if access.intersects(
            Access::FragmentSampled
                | Access::VertexSampled
                | Access::ComputeSampled
                | Access::RayTracingSampled
                | Access::TessellationSampled,
        ) {
            return vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        }
        if access.intersects(
            Access::FragmentRead
                | Access::FragmentWrite
                | Access::FragmentRW
                | Access::ComputeRead
                | Access::ComputeWrite
                | Access::RayTracingRead
                | Access::RayTracingWrite,
        ) {
            return vk::ImageLayout::GENERAL;
        }
        if access.intersects(Access::TransferWrite | Access::CopyWrite | Access::BlitWrite | Access::Clear) {
            return vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        }
        if access.intersects(Access::TransferRead | Access::CopyRead | Access::BlitRead) {
            return vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }

        vk::ImageLayout::UNDEFINED
    }
}

impl From<Access> for vk::BufferUsageFlags {
    fn from(access: Access) -> Self {
        let mut usage = vk::BufferUsageFlags::empty();

        if access.intersects(Access::MemoryRW | Access::FragmentRW | Access::ComputeRW | Access::RayTracingRW) {
            usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if access.intersects(
            Access::MemoryRW
                | Access::ComputeUniformRead
                | Access::VertexUniformRead
                | Access::FragmentUniformRead
                | Access::TessellationUniformRead
                | Access::RayTracingUniformRead,
        ) {
            usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if access.intersects(Access::MemoryRW | Access::AttributeRead) {
            usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if access.intersects(Access::MemoryRW | Access::IndexRead) {
            usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if access.intersects(Access::MemoryRW | Access::IndirectRead) {
            usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        if access.intersects(Access::MemoryRW | Access::TransferRead) {
            usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if access.intersects(Access::MemoryRW | Access::TransferWrite) {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        usage
    }
}

impl From<Access> for vk::ImageUsageFlags {
    fn from(access: Access) -> Self {
        let mut usage = vk::ImageUsageFlags::empty();

        if access.intersects(Access::MemoryRW | Access::ColorRW) {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if access.intersects(
            Access::MemoryRW
                | Access::FragmentSampled
                | Access::ComputeSampled
                | Access::RayTracingSampled
                | Access::VertexSampled
                | Access::TessellationSampled,
        ) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if access.intersects(Access::MemoryRW | Access::DepthStencilRW) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if access.intersects(Access::MemoryRW | Access::TransferRead) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if access.intersects(Access::MemoryRW | Access::TransferWrite) {
            usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if access.intersects(Access::MemoryRW | Access::FragmentRW | Access::ComputeRW | Access::RayTracingRW) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }

        usage
    }
}
