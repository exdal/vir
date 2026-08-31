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
        const VertexWrite = 1 << 41;
        const VertexRW = Self::VertexRead.bits() | Self::VertexWrite.bits();
        const TessellationControlSampled = 1 << 42;
        const TessellationControlRead = 1 << 43;
        const TessellationControlWrite = 1 << 44;
        const TessellationControlRW = Self::TessellationControlRead.bits() | Self::TessellationControlWrite.bits();
        const TessellationControlUniformRead = 1 << 45;
        const TessellationEvaluationSampled = 1 << 46;
        const TessellationEvaluationRead = 1 << 47;
        const TessellationEvaluationWrite = 1 << 48;
        const TessellationEvaluationRW = Self::TessellationEvaluationRead.bits() | Self::TessellationEvaluationWrite.bits();
        const TessellationEvaluationUniformRead = 1 << 49;
        const GeometrySampled = 1 << 50;
        const GeometryRead = 1 << 51;
        const GeometryWrite = 1 << 52;
        const GeometryRW = Self::GeometryRead.bits() | Self::GeometryWrite.bits();
        const GeometryUniformRead = 1 << 53;
        const InputAttachmentRead = 1 << 54;
        const VertexAccelerationStructureRead = 1 << 55;
        const TessellationControlAccelerationStructureRead = 1 << 56;
        const TessellationEvaluationAccelerationStructureRead = 1 << 57;
        const GeometryAccelerationStructureRead = 1 << 58;
        const FragmentAccelerationStructureRead = 1 << 59;
        const ComputeAccelerationStructureRead = 1 << 60;
        const Writes = Self::ColorWrite.bits()
            | Self::DepthStencilWrite.bits()
            | Self::VertexWrite.bits()
            | Self::TessellationControlWrite.bits()
            | Self::TessellationEvaluationWrite.bits()
            | Self::GeometryWrite.bits()
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

    pub fn sampled_by(stages: vk::ShaderStageFlags) -> Access {
        let mapped = [
            (vk::ShaderStageFlags::VERTEX, Access::VertexSampled),
            (vk::ShaderStageFlags::FRAGMENT, Access::FragmentSampled),
            (vk::ShaderStageFlags::COMPUTE, Access::ComputeSampled),
            (
                vk::ShaderStageFlags::TESSELLATION_CONTROL,
                Access::TessellationControlSampled,
            ),
            (
                vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                Access::TessellationEvaluationSampled,
            ),
            (vk::ShaderStageFlags::GEOMETRY, Access::GeometrySampled),
            (vk::ShaderStageFlags::RAYGEN_KHR, Access::RayTracingSampled),
            (vk::ShaderStageFlags::ANY_HIT_KHR, Access::RayTracingSampled),
            (vk::ShaderStageFlags::CLOSEST_HIT_KHR, Access::RayTracingSampled),
            (vk::ShaderStageFlags::MISS_KHR, Access::RayTracingSampled),
            (vk::ShaderStageFlags::INTERSECTION_KHR, Access::RayTracingSampled),
            (vk::ShaderStageFlags::CALLABLE_KHR, Access::RayTracingSampled),
        ];

        let mut access = Access::empty();
        let mut covered = vk::ShaderStageFlags::empty();
        for (stage, sampled) in mapped {
            if stages.contains(stage) {
                access |= sampled;
                covered |= stage;
            }
        }

        let uncovered = stages & !covered;
        if !uncovered.is_empty() {
            tracing::warn!(stages = ?uncovered, "no sampled access covers these stages; memory is waited on instead");
            access |= Access::MemoryRead;
        }

        access
    }

    pub(crate) fn shader_read_by(stage: vk::ShaderStageFlags) -> Access {
        Self::by_shader_stage(
            stage,
            Access::VertexRead,
            Access::TessellationControlRead,
            Access::TessellationEvaluationRead,
            Access::GeometryRead,
            Access::FragmentRead,
            Access::ComputeRead,
            Access::RayTracingRead,
            Access::MemoryRead,
        )
    }

    pub(crate) fn shader_write_by(stage: vk::ShaderStageFlags) -> Access {
        Self::by_shader_stage(
            stage,
            Access::VertexWrite,
            Access::TessellationControlWrite,
            Access::TessellationEvaluationWrite,
            Access::GeometryWrite,
            Access::FragmentWrite,
            Access::ComputeWrite,
            Access::RayTracingWrite,
            Access::MemoryWrite,
        )
    }

    pub(crate) fn uniform_by(stage: vk::ShaderStageFlags) -> Access {
        Self::by_shader_stage(
            stage,
            Access::VertexUniformRead,
            Access::TessellationControlUniformRead,
            Access::TessellationEvaluationUniformRead,
            Access::GeometryUniformRead,
            Access::FragmentUniformRead,
            Access::ComputeUniformRead,
            Access::RayTracingUniformRead,
            Access::MemoryRead,
        )
    }

    pub(crate) fn acceleration_structure_read_by(stage: vk::ShaderStageFlags) -> Access {
        Self::by_shader_stage(
            stage,
            Access::VertexAccelerationStructureRead,
            Access::TessellationControlAccelerationStructureRead,
            Access::TessellationEvaluationAccelerationStructureRead,
            Access::GeometryAccelerationStructureRead,
            Access::FragmentAccelerationStructureRead,
            Access::ComputeAccelerationStructureRead,
            Access::RayTracingRead,
            Access::MemoryRead,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn by_shader_stage(
        stages: vk::ShaderStageFlags, vertex: Access, tessellation_control: Access, tessellation_evaluation: Access,
        geometry: Access, fragment: Access, compute: Access, ray_tracing: Access, fallback: Access,
    ) -> Access {
        let mapped = [
            (vk::ShaderStageFlags::VERTEX, vertex),
            (vk::ShaderStageFlags::TESSELLATION_CONTROL, tessellation_control),
            (vk::ShaderStageFlags::TESSELLATION_EVALUATION, tessellation_evaluation),
            (vk::ShaderStageFlags::GEOMETRY, geometry),
            (vk::ShaderStageFlags::FRAGMENT, fragment),
            (vk::ShaderStageFlags::COMPUTE, compute),
            (vk::ShaderStageFlags::RAYGEN_KHR, ray_tracing),
            (vk::ShaderStageFlags::ANY_HIT_KHR, ray_tracing),
            (vk::ShaderStageFlags::CLOSEST_HIT_KHR, ray_tracing),
            (vk::ShaderStageFlags::MISS_KHR, ray_tracing),
            (vk::ShaderStageFlags::INTERSECTION_KHR, ray_tracing),
            (vk::ShaderStageFlags::CALLABLE_KHR, ray_tracing),
        ];

        let mut access = Access::empty();
        let mut covered = vk::ShaderStageFlags::empty();
        for (stage, mapped_access) in mapped {
            if stages.contains(stage) {
                access |= mapped_access;
                covered |= stage;
            }
        }

        let uncovered = stages & !covered;
        if !uncovered.is_empty() {
            tracing::warn!(stages = ?uncovered, "no access covers these shader stages; memory is waited on instead");
            access |= fallback;
        }

        access
    }
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
        if access.contains(Access::VertexWrite) {
            flags |= vk::AccessFlags2::SHADER_WRITE;
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
        if access.intersects(
            Access::TessellationControlSampled | Access::TessellationEvaluationSampled | Access::GeometrySampled,
        ) {
            flags |= vk::AccessFlags2::SHADER_SAMPLED_READ;
        }
        if access
            .intersects(Access::TessellationControlRead | Access::TessellationEvaluationRead | Access::GeometryRead)
        {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if access
            .intersects(Access::TessellationControlWrite | Access::TessellationEvaluationWrite | Access::GeometryWrite)
        {
            flags |= vk::AccessFlags2::SHADER_WRITE;
        }
        if access.intersects(
            Access::TessellationControlUniformRead
                | Access::TessellationEvaluationUniformRead
                | Access::GeometryUniformRead,
        ) {
            flags |= vk::AccessFlags2::UNIFORM_READ;
        }
        if access.contains(Access::InputAttachmentRead) {
            flags |= vk::AccessFlags2::INPUT_ATTACHMENT_READ;
        }
        if access.intersects(
            Access::VertexAccelerationStructureRead
                | Access::TessellationControlAccelerationStructureRead
                | Access::TessellationEvaluationAccelerationStructureRead
                | Access::GeometryAccelerationStructureRead
                | Access::FragmentAccelerationStructureRead
                | Access::ComputeAccelerationStructureRead,
        ) {
            flags |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
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
        if access.contains(Access::VertexWrite) {
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
        if access.intersects(
            Access::TessellationControlSampled
                | Access::TessellationControlRead
                | Access::TessellationControlWrite
                | Access::TessellationControlUniformRead
                | Access::TessellationControlAccelerationStructureRead,
        ) {
            flags |= vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER;
        }
        if access.intersects(
            Access::TessellationEvaluationSampled
                | Access::TessellationEvaluationRead
                | Access::TessellationEvaluationWrite
                | Access::TessellationEvaluationUniformRead
                | Access::TessellationEvaluationAccelerationStructureRead,
        ) {
            flags |= vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER;
        }
        if access.intersects(
            Access::GeometrySampled
                | Access::GeometryRead
                | Access::GeometryWrite
                | Access::GeometryUniformRead
                | Access::GeometryAccelerationStructureRead,
        ) {
            flags |= vk::PipelineStageFlags2::GEOMETRY_SHADER;
        }
        if access.contains(Access::InputAttachmentRead) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if access.contains(Access::VertexAccelerationStructureRead) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if access.contains(Access::FragmentAccelerationStructureRead) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if access.contains(Access::ComputeAccelerationStructureRead) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
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
                | Access::TessellationSampled
                | Access::TessellationControlSampled
                | Access::TessellationEvaluationSampled
                | Access::GeometrySampled
                | Access::InputAttachmentRead,
        ) {
            return vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        }
        if access.intersects(
            Access::FragmentRead
                | Access::FragmentWrite
                | Access::FragmentRW
                | Access::VertexRead
                | Access::VertexWrite
                | Access::TessellationControlRead
                | Access::TessellationControlWrite
                | Access::TessellationEvaluationRead
                | Access::TessellationEvaluationWrite
                | Access::GeometryRead
                | Access::GeometryWrite
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

        if access.intersects(
            Access::MemoryRW
                | Access::VertexRW
                | Access::TessellationControlRW
                | Access::TessellationEvaluationRW
                | Access::GeometryRW
                | Access::FragmentRW
                | Access::ComputeRW
                | Access::RayTracingRW,
        ) {
            usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if access.intersects(
            Access::MemoryRW
                | Access::ComputeUniformRead
                | Access::VertexUniformRead
                | Access::FragmentUniformRead
                | Access::TessellationUniformRead
                | Access::TessellationControlUniformRead
                | Access::TessellationEvaluationUniformRead
                | Access::GeometryUniformRead
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
                | Access::TessellationSampled
                | Access::TessellationControlSampled
                | Access::TessellationEvaluationSampled
                | Access::GeometrySampled,
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
        if access.intersects(
            Access::MemoryRW
                | Access::VertexRW
                | Access::TessellationControlRW
                | Access::TessellationEvaluationRW
                | Access::GeometryRW
                | Access::FragmentRW
                | Access::ComputeRW
                | Access::RayTracingRW,
        ) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if access.intersects(Access::MemoryRW | Access::InputAttachmentRead) {
            usage |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
        }

        usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_access_keeps_each_graphics_stage_distinct() {
        let stages = vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::TESSELLATION_CONTROL
            | vk::ShaderStageFlags::TESSELLATION_EVALUATION
            | vk::ShaderStageFlags::GEOMETRY
            | vk::ShaderStageFlags::FRAGMENT;
        assert_eq!(
            Access::sampled_by(stages),
            Access::VertexSampled
                | Access::TessellationControlSampled
                | Access::TessellationEvaluationSampled
                | Access::GeometrySampled
                | Access::FragmentSampled
        );
    }

    #[test]
    fn compute_uniform_reads_convert_to_the_matching_vulkan_access_and_stage() {
        assert_eq!(
            vk::AccessFlags2::from(Access::ComputeUniformRead),
            vk::AccessFlags2::UNIFORM_READ
        );
        assert_eq!(
            vk::PipelineStageFlags2::from(Access::ComputeUniformRead),
            vk::PipelineStageFlags2::COMPUTE_SHADER
        );
    }

    #[test]
    fn geometry_storage_writes_are_writes_at_the_geometry_stage() {
        assert!(Access::GeometryWrite.writes());
        assert_eq!(
            vk::AccessFlags2::from(Access::GeometryWrite),
            vk::AccessFlags2::SHADER_WRITE
        );
        assert_eq!(
            vk::PipelineStageFlags2::from(Access::GeometryWrite),
            vk::PipelineStageFlags2::GEOMETRY_SHADER
        );
    }

    #[test]
    fn input_attachment_access_selects_its_layout_and_usage() {
        assert_eq!(
            vk::ImageLayout::from(Access::InputAttachmentRead),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            vk::ImageUsageFlags::from(Access::InputAttachmentRead),
            vk::ImageUsageFlags::INPUT_ATTACHMENT
        );
    }
}
