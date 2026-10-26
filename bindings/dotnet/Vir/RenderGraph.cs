namespace Vir;

public sealed unsafe class RenderGraph : IDisposable
{
    private nint handle;

    public RenderGraph(Context context)
    {
        nint handle;
        VirException.ThrowIfFailed(PInvoke.RenderGraphCreate(context.Handle, &handle));
        this.handle = handle;
    }

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(RenderGraph));

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        PInvoke.RenderGraphDestroy(this.handle);
        this.handle = 0;
    }

    public PipelineId DeclareGraphicsPipeline(
        ReadOnlySpan<byte> vertexSpirv,
        ReadOnlySpan<byte> fragmentSpirv,
        BindlessSet? bindless = null
    )
    {
        uint pipeline;
        var native = bindless is { } set ? set.ToNative() : default;
        var nativePtr = bindless.HasValue ? &native : null;
        fixed (byte* vertex = vertexSpirv)
        fixed (byte* fragment = fragmentSpirv)
        {
            VirException.ThrowIfFailed(
                nativePtr is null
                    ? PInvoke.RenderGraphDeclareGraphicsPipeline(
                        this.Handle,
                        vertex,
                        (nuint)vertexSpirv.Length,
                        fragment,
                        (nuint)fragmentSpirv.Length,
                        &pipeline
                    )
                    : PInvoke.RenderGraphDeclareGraphicsPipelineBindless(
                        this.Handle,
                        vertex,
                        (nuint)vertexSpirv.Length,
                        fragment,
                        (nuint)fragmentSpirv.Length,
                        nativePtr,
                        &pipeline
                    )
            );
        }

        return new PipelineId(pipeline);
    }

    public PipelineId DeclareComputePipeline(ReadOnlySpan<byte> spirv, BindlessSet? bindless = null)
    {
        uint pipeline;
        var native = bindless is { } set ? set.ToNative() : default;
        var nativePtr = bindless.HasValue ? &native : null;
        fixed (byte* ptr = spirv)
        {
            VirException.ThrowIfFailed(
                nativePtr is null
                    ? PInvoke.RenderGraphDeclareComputePipeline(this.Handle, ptr, (nuint)spirv.Length, &pipeline)
                    : PInvoke.RenderGraphDeclareComputePipelineBindless(this.Handle, ptr, (nuint)spirv.Length, nativePtr, &pipeline)
            );
        }

        return new PipelineId(pipeline);
    }

    public bool Execute(Context context, Program program, FrameAllocator frame)
    {
        var result = PInvoke.RenderGraphExecute(this.Handle, context.Handle, program.Handle, frame.Handle);
        if (result == VkResult.ErrorOutOfDate)
        {
            VirException.TakeLastError();
            return false;
        }

        VirException.ThrowIfFailed(result);
        return true;
    }

    public void ExecuteBlocking(Context context, Program program, PersistentAllocator allocator) =>
        VirException.ThrowIfFailed(
            PInvoke.RenderGraphExecuteBlocking(this.Handle, context.Handle, program.Handle, allocator.Handle)
        );

    public void Wait() => VirException.ThrowIfFailed(PInvoke.RenderGraphWait(this.Handle));
}
