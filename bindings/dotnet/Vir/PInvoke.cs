using System.Runtime.InteropServices;

namespace Vir;

[StructLayout(LayoutKind.Sequential)]
internal struct VirAttachment
{
    public uint Value;
    public ulong Access;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirRect2D
{
    public uint Tag;
    public int X;
    public int Y;
    public uint Width;
    public uint Height;
    public float RelativeX;
    public float RelativeY;
    public float RelativeWidth;
    public float RelativeHeight;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirViewport
{
    public VirRect2D Rect;
    public float MinDepth;
    public float MaxDepth;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirRasterizationState
{
    public int PolygonMode;
    public uint CullMode;
    public int FrontFace;
    public float LineWidth;
    public uint DepthClampEnable;
    public uint RasterizerDiscardEnable;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirClearValue
{
    public ulong Low;
    public ulong High;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirContextCreateInfo
{
    public nint GetInstanceProcAddr;
    public nint Instance;
    public nint PhysicalDevice;
    public nint Device;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VirSwapChainInfo
{
    public ulong SwapChain;
    public ulong* Images;
    public uint ImageCount;
    public int Format;
    public uint Width;
    public uint Height;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirDepthState
{
    public uint TestEnable;
    public uint WriteEnable;
    public int CompareOp;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VirBufferInfo
{
    public ulong Size;
    public uint Usage;
    public uint Location;
    public ulong Alignment;
    public byte* Name;
    public nuint NameLength;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VirImageInfo
{
    public uint Kind;
    public uint Width;
    public uint Height;
    public int Format;
    public uint Usage;
    public uint MipLevels;
    public uint ArrayLayers;
    public uint Samples;
    public byte* Name;
    public nuint NameLength;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VirTransientImageInfo
{
    public uint Width;
    public uint Height;
    public uint Depth;
    public int Format;
    public uint Usage;
    public uint MipLevels;
    public uint ArrayLayers;
    public uint Samples;
    public byte* Name;
    public nuint NameLength;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirBindlessSet
{
    public uint Index;
    public ulong Layout;
    public ulong Set;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirBufferCopy
{
    public ulong SrcOffset;
    public ulong DstOffset;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirBufferImageCopy
{
    public ulong BufferOffset;
    public uint MipLevel;
    public int OffsetX;
    public int OffsetY;
    public int OffsetZ;
    public uint Width;
    public uint Height;
    public uint Depth;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VirSamplerInfo
{
    public int MagFilter;
    public int MinFilter;
    public int MipmapMode;
    public int AddressModeU;
    public int AddressModeV;
    public int AddressModeW;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VirPassTiming
{
    public byte* Name;
    public nuint NameLength;
    public ulong DurationNs;
}

// TODO: theres gotta be a better way of doing this, this is my first time writing FFI
internal static unsafe class NativeLayout
{
    public static bool IsValid() =>
        sizeof(VirAttachment) == 16
        && sizeof(VirRect2D) == 36
        && sizeof(VirViewport) == 44
        && sizeof(VirRasterizationState) == 24
        && sizeof(VirClearValue) == 16
        && sizeof(VirContextCreateInfo) == 32
        && sizeof(VirSwapChainInfo) == 32
        && sizeof(VirDepthState) == 12
        && sizeof(VirBufferInfo) == 40
        && sizeof(VirImageInfo) == 48
        && sizeof(VirTransientImageInfo) == 48
        && sizeof(VirSamplerInfo) == 24
        && sizeof(VirBindlessSet) == 24
        && sizeof(VirBufferCopy) == 24
        && sizeof(VirBufferImageCopy) == 40
        && sizeof(VirPassTiming) == 24;
}

internal static unsafe partial class PInvoke
{
    private const string Dll = Loader.LogicalName;

    [LibraryImport(Dll, EntryPoint = "vir_last_error")]
    internal static partial nuint LastError(byte* buffer, nuint length);

    [LibraryImport(Dll, EntryPoint = "vir_clear_last_error")]
    internal static partial void ClearLastError();

    [LibraryImport(Dll, EntryPoint = "vir_module_create")]
    internal static partial nint ModuleCreate();

    [LibraryImport(Dll, EntryPoint = "vir_module_destroy")]
    internal static partial void ModuleDestroy(nint module);

    [LibraryImport(Dll, EntryPoint = "vir_module_declare_bool_var")]
    internal static partial uint ModuleDeclareBoolVar(nint module, byte* name, nuint nameLength, byte value);

    [LibraryImport(Dll, EntryPoint = "vir_module_declare_bytes_var")]
    internal static partial uint ModuleDeclareBytesVar(nint module, byte* name, nuint nameLength, uint size);

    [LibraryImport(Dll, EntryPoint = "vir_module_clear")]
    internal static partial uint ModuleClear(nint module, uint attachment, VirClearValue color);

    [LibraryImport(Dll, EntryPoint = "vir_module_begin_rendering")]
    internal static partial void ModuleBeginRendering(nint module, VirAttachment* attachments, nuint count);

    [LibraryImport(Dll, EntryPoint = "vir_module_with_name")]
    internal static partial void ModuleWithName(nint module, byte* name, nuint nameLength);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_graphics_pipeline")]
    internal static partial void ModuleBindGraphicsPipeline(nint module, uint pipeline);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_viewport")]
    internal static partial void ModuleSetViewport(nint module, uint index, VirViewport viewport);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_scissor")]
    internal static partial void ModuleSetScissor(nint module, uint index, VirRect2D rect);

    [LibraryImport(Dll, EntryPoint = "vir_module_broadcast_color_blend")]
    internal static partial void ModuleBroadcastColorBlend(nint module, uint preset);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_rasterization")]
    internal static partial void ModuleSetRasterization(nint module, VirRasterizationState state);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_dynamic_state")]
    internal static partial void ModuleSetDynamicState(nint module, uint flags);

    [LibraryImport(Dll, EntryPoint = "vir_module_push_constant_bytes")]
    internal static partial void ModulePushConstantBytes(nint module, uint offset, byte* data, nuint length);

    [LibraryImport(Dll, EntryPoint = "vir_module_push_constants_from_at")]
    internal static partial void ModulePushConstantsFromAt(nint module, uint offset, uint variable);

    [LibraryImport(Dll, EntryPoint = "vir_module_draw")]
    internal static partial void ModuleDraw(nint module, uint vertexCount, uint instanceCount);

    [LibraryImport(Dll, EntryPoint = "vir_module_end_rendering")]
    internal static partial VkResult ModuleEndRendering(nint module, uint* output, nuint count);

    [LibraryImport(Dll, EntryPoint = "vir_module_present")]
    internal static partial uint ModulePresent(nint module, uint attachment);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_condition")]
    internal static partial VkResult ModuleSetCondition(
        nint module,
        uint condition,
        delegate* unmanaged[Cdecl]<nint, nint, uint*, nuint, nuint> then,
        delegate* unmanaged[Cdecl]<nint, nint, uint*, nuint, nuint> otherwise,
        nint userData,
        uint* output,
        nuint capacity,
        nuint* count
    );

    [LibraryImport(Dll, EntryPoint = "vir_program_destroy")]
    internal static partial void ProgramDestroy(nint program);

    [LibraryImport(Dll, EntryPoint = "vir_program_set_bool")]
    internal static partial VkResult ProgramSetBool(nint program, uint variable, byte value);

    [LibraryImport(Dll, EntryPoint = "vir_program_set_bytes")]
    internal static partial VkResult ProgramSetBytes(nint program, uint variable, byte* data, nuint length);

    [LibraryImport(Dll, EntryPoint = "vir_context_create")]
    internal static partial VkResult ContextCreate(VirContextCreateInfo* info, nint* context);

    [LibraryImport(Dll, EntryPoint = "vir_context_destroy")]
    internal static partial void ContextDestroy(nint context);

    [LibraryImport(Dll, EntryPoint = "vir_context_create_command_queue")]
    internal static partial VkResult ContextCreateCommandQueue(nint context, uint queueFamilyIndex, uint domain);

    [LibraryImport(Dll, EntryPoint = "vir_context_wait_idle")]
    internal static partial VkResult ContextWaitIdle(nint context);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_create")]
    internal static partial VkResult PersistentAllocatorCreate(nint context, nint* allocator);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_destroy")]
    internal static partial void PersistentAllocatorDestroy(nint allocator);

    [LibraryImport(Dll, EntryPoint = "vir_super_frame_allocator_create")]
    internal static partial VkResult SuperFrameAllocatorCreate(nint context, uint frameCount, nint* allocator);

    [LibraryImport(Dll, EntryPoint = "vir_super_frame_allocator_destroy")]
    internal static partial void SuperFrameAllocatorDestroy(nint allocator);

    [LibraryImport(Dll, EntryPoint = "vir_super_frame_allocator_next_frame")]
    internal static partial VkResult SuperFrameAllocatorNextFrame(nint allocator, nint* frame);

    [LibraryImport(Dll, EntryPoint = "vir_super_frame_allocator_enable_timestamps")]
    internal static partial VkResult SuperFrameAllocatorEnableTimestamps(nint allocator, nint context, uint* enabled);

    [LibraryImport(Dll, EntryPoint = "vir_frame_allocator_completed_timings")]
    internal static partial VkResult FrameAllocatorCompletedTimings(nint frame, VirPassTiming* timings, nuint capacity, nuint* count);

    [LibraryImport(Dll, EntryPoint = "vir_frame_allocator_add_timeline_wait")]
    internal static partial void FrameAllocatorAddTimelineWait(nint frame, ulong semaphore, ulong value);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_create")]
    internal static partial VkResult RenderGraphCreate(nint context, nint* graph);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_destroy")]
    internal static partial void RenderGraphDestroy(nint graph);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_declare_graphics_pipeline")]
    internal static partial VkResult RenderGraphDeclareGraphicsPipeline(
        nint graph,
        byte* vertex,
        nuint vertexLength,
        byte* fragment,
        nuint fragmentLength,
        uint* pipeline
    );

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_execute")]
    internal static partial VkResult RenderGraphExecute(nint graph, nint context, nint program, nint frame);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_execute_blocking")]
    internal static partial VkResult RenderGraphExecuteBlocking(nint graph, nint context, nint program, nint allocator);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_wait")]
    internal static partial VkResult RenderGraphWait(nint graph);

    [LibraryImport(Dll, EntryPoint = "vir_swapchain_create")]
    internal static partial VkResult SwapChainCreate(nint allocator, VirSwapChainInfo* info, nint* swapchain);

    [LibraryImport(Dll, EntryPoint = "vir_swapchain_destroy")]
    internal static partial void SwapChainDestroy(nint context, nint swapchain);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_allocate_buffer")]
    internal static partial VkResult PersistentAllocatorAllocateBuffer(nint allocator, VirBufferInfo* info, nint* buffer);

    [LibraryImport(Dll, EntryPoint = "vir_buffer_write")]
    internal static partial VkResult BufferWrite(nint buffer, ulong offset, byte* data, nuint length);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_deallocate_buffer")]
    internal static partial void PersistentAllocatorDeallocateBuffer(nint allocator, nint buffer);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_allocate_image")]
    internal static partial VkResult PersistentAllocatorAllocateImage(nint allocator, VirImageInfo* info, nint* image);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_deallocate_image")]
    internal static partial void PersistentAllocatorDeallocateImage(nint allocator, nint image);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_allocate_sampler")]
    internal static partial VkResult PersistentAllocatorAllocateSampler(nint allocator, VirSamplerInfo* info, ulong* sampler);

    [LibraryImport(Dll, EntryPoint = "vir_persistent_allocator_deallocate_sampler")]
    internal static partial void PersistentAllocatorDeallocateSampler(nint allocator, ulong sampler);

    [LibraryImport(Dll, EntryPoint = "vir_module_acquire_next_image")]
    internal static partial uint ModuleAcquireNextImage(nint module, nint swapchain);

    [LibraryImport(Dll, EntryPoint = "vir_module_compile")]
    internal static partial VkResult ModuleCompile(nint module, nint graph, uint root, nint* program);

    [LibraryImport(Dll, EntryPoint = "vir_module_compile_all")]
    internal static partial VkResult ModuleCompileAll(nint module, nint graph, uint* roots, nuint count, nint* program);

    [LibraryImport(Dll, EntryPoint = "vir_access_image_layout")]
    internal static partial int AccessImageLayout(ulong access);

    [LibraryImport(Dll, EntryPoint = "vir_module_import_buffer")]
    internal static partial uint ModuleImportBuffer(nint module, nint buffer, ulong access);

    [LibraryImport(Dll, EntryPoint = "vir_module_import_image")]
    internal static partial uint ModuleImportImage(nint module, nint image, int layout, ulong access);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_name")]
    internal static partial uint ModuleSetName(nint module, uint value, byte* name, nuint nameLength);

    [LibraryImport(Dll, EntryPoint = "vir_module_copy_buffer_to_image")]
    internal static partial uint ModuleCopyBufferToImage(nint module, uint buffer, uint image);

    [LibraryImport(Dll, EntryPoint = "vir_module_export")]
    internal static partial uint ModuleExport(nint module, uint value, ulong access, uint domain);

    [LibraryImport(Dll, EntryPoint = "vir_module_set_depth")]
    internal static partial void ModuleSetDepth(nint module, VirDepthState depth);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_texture")]
    internal static partial void ModuleBindTexture(nint module, uint set, uint binding, uint image, ulong sampler);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_vertex_buffer")]
    internal static partial void ModuleBindVertexBuffer(nint module, uint binding, uint buffer);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_index_buffer")]
    internal static partial void ModuleBindIndexBuffer(nint module, uint buffer, int indexType);

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_indexed")]
    internal static partial void ModuleDrawIndexed(nint module, uint indexCount, uint instanceCount);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_declare_compute_pipeline")]
    internal static partial VkResult RenderGraphDeclareComputePipeline(nint graph, byte* spirv, nuint spirvLength, uint* pipeline);

    [LibraryImport(Dll, EntryPoint = "vir_module_begin_compute")]
    internal static partial void ModuleBeginCompute(nint module, VirAttachment* attachments, nuint count);

    [LibraryImport(Dll, EntryPoint = "vir_module_end_compute")]
    internal static partial VkResult ModuleEndCompute(nint module, uint* output, nuint count);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_compute_pipeline")]
    internal static partial void ModuleBindComputePipeline(nint module, uint pipeline);

    [LibraryImport(Dll, EntryPoint = "vir_module_dispatch")]
    internal static partial void ModuleDispatch(nint module, uint x, uint y, uint z);

    [LibraryImport(Dll, EntryPoint = "vir_module_dispatch_invocations")]
    internal static partial void ModuleDispatchInvocations(nint module, uint x, uint y, uint z);

    [LibraryImport(Dll, EntryPoint = "vir_module_dispatch_invocations_per_pixel_scaled")]
    internal static partial void ModuleDispatchInvocationsPerPixelScaled(nint module, uint image, float scaleX, float scaleY, float scaleZ);

    [LibraryImport(Dll, EntryPoint = "vir_module_dispatch_invocations_per_element_scaled")]
    internal static partial void ModuleDispatchInvocationsPerElementScaled(nint module, uint buffer, ulong elementSize, float scale);

    [LibraryImport(Dll, EntryPoint = "vir_module_dispatch_indirect_at")]
    internal static partial void ModuleDispatchIndirectAt(nint module, uint buffer, ulong offset);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_image")]
    internal static partial void ModuleBindImage(nint module, uint set, uint binding, uint image);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_buffer_range")]
    internal static partial void ModuleBindBufferRange(nint module, uint set, uint binding, uint buffer, ulong offset, ulong range);

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_sampler")]
    internal static partial void ModuleBindSampler(nint module, uint set, uint binding, ulong sampler);

    [LibraryImport(Dll, EntryPoint = "vir_module_push_constant_address")]
    internal static partial void ModulePushConstantAddress(nint module, uint offset, uint buffer);

    [LibraryImport(Dll, EntryPoint = "vir_module_specialize_constant")]
    internal static partial void ModuleSpecializeConstant(nint module, uint id, uint kind, ulong bits);

    [LibraryImport(Dll, EntryPoint = "vir_module_transient_image")]
    internal static partial uint ModuleTransientImage(nint module, VirTransientImageInfo* info);

    [LibraryImport(Dll, EntryPoint = "vir_module_transient_buffer")]
    internal static partial uint ModuleTransientBuffer(nint module, VirBufferInfo* info);

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_declare_graphics_pipeline_bindless")]
    internal static partial VkResult RenderGraphDeclareGraphicsPipelineBindless(
        nint graph,
        byte* vertex,
        nuint vertexLength,
        byte* fragment,
        nuint fragmentLength,
        VirBindlessSet* bindless,
        uint* pipeline
    );

    [LibraryImport(Dll, EntryPoint = "vir_render_graph_declare_compute_pipeline_bindless")]
    internal static partial VkResult RenderGraphDeclareComputePipelineBindless(
        nint graph,
        byte* spirv,
        nuint spirvLength,
        VirBindlessSet* bindless,
        uint* pipeline
    );

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_range")]
    internal static partial void ModuleDrawRange(nint module, uint vertexCount, uint instanceCount, uint firstVertex, uint firstInstance);

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_indexed_range")]
    internal static partial void ModuleDrawIndexedRange(
        nint module,
        uint indexCount,
        uint instanceCount,
        uint firstIndex,
        int vertexOffset,
        uint firstInstance
    );

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_indirect_at")]
    internal static partial void ModuleDrawIndirectAt(nint module, uint buffer, ulong offset, uint drawCount, uint stride);

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_indexed_indirect_at")]
    internal static partial void ModuleDrawIndexedIndirectAt(nint module, uint buffer, ulong offset, uint drawCount, uint stride);

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_indirect_count_at")]
    internal static partial void ModuleDrawIndirectCountAt(
        nint module,
        uint buffer,
        ulong offset,
        uint countBuffer,
        ulong countOffset,
        uint maxDrawCount,
        uint stride
    );

    [LibraryImport(Dll, EntryPoint = "vir_module_draw_indexed_indirect_count_at")]
    internal static partial void ModuleDrawIndexedIndirectCountAt(
        nint module,
        uint buffer,
        ulong offset,
        uint countBuffer,
        ulong countOffset,
        uint maxDrawCount,
        uint stride
    );

    [LibraryImport(Dll, EntryPoint = "vir_module_bind_index_buffer_at")]
    internal static partial void ModuleBindIndexBufferAt(nint module, uint buffer, ulong offset, int indexType);

    [LibraryImport(Dll, EntryPoint = "vir_module_copy_buffer")]
    internal static partial uint ModuleCopyBuffer(nint module, uint src, uint dst, VirBufferCopy* regions, nuint count);

    [LibraryImport(Dll, EntryPoint = "vir_module_copy_buffer_to_image_region")]
    internal static partial uint ModuleCopyBufferToImageRegion(nint module, uint buffer, uint image, VirBufferImageCopy* regions, nuint count);

    [LibraryImport(Dll, EntryPoint = "vir_module_declare_buffer_var")]
    internal static partial uint ModuleDeclareBufferVar(nint module, byte* name, nuint nameLength, ulong access);

    [LibraryImport(Dll, EntryPoint = "vir_module_declare_u32_var")]
    internal static partial uint ModuleDeclareU32Var(nint module, byte* name, nuint nameLength, uint value);

    [LibraryImport(Dll, EntryPoint = "vir_program_set_u32")]
    internal static partial VkResult ProgramSetU32(nint program, uint variable, uint value);

    [LibraryImport(Dll, EntryPoint = "vir_program_set_buffer")]
    internal static partial VkResult ProgramSetBuffer(nint program, uint variable, nint buffer);

    [LibraryImport(Dll, EntryPoint = "vir_buffer_device_address")]
    internal static partial ulong BufferDeviceAddress(nint buffer);

    [LibraryImport(Dll, EntryPoint = "vir_buffer_size")]
    internal static partial ulong BufferSize(nint buffer);

    [LibraryImport(Dll, EntryPoint = "vir_image_view_handle")]
    internal static partial ulong ImageViewHandle(nint image);

    [LibraryImport(Dll, EntryPoint = "vir_image_handle")]
    internal static partial ulong ImageHandle(nint image);
}
