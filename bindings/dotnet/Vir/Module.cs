using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Vir;

public sealed unsafe class Module : IDisposable
{
    private const int StackLimit = 256;

    private nint handle;
    private readonly bool isOwner;

    public Module()
    {
        this.handle = PInvoke.ModuleCreate();
        if (this.handle == 0)
            throw new VirException(VkResult.ErrorUnknown, "failed to create module");

        this.isOwner = true;
    }

    private Module(nint handle)
    {
        this.handle = handle;
        this.isOwner = false;
    }

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(Module));

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        if (this.isOwner)
            PInvoke.ModuleDestroy(this.handle);

        this.handle = 0;
    }

    public ValueId DeclareBoolVar(string name, bool value)
    {
        var utf8 = Utf8Buffer.Encode(name, stackalloc byte[StackLimit]);
        fixed (byte* ptr = utf8)
            return VirException.CheckId(PInvoke.ModuleDeclareBoolVar(this.Handle, ptr, (nuint)utf8.Length, value ? (byte)1 : (byte)0));
    }

    public ValueId DeclareBytesVar(string name, uint size)
    {
        var utf8 = Utf8Buffer.Encode(name, stackalloc byte[StackLimit]);
        fixed (byte* ptr = utf8)
            return VirException.CheckId(PInvoke.ModuleDeclareBytesVar(this.Handle, ptr, (nuint)utf8.Length, size));
    }

    public ValueId DeclareBytesVar<T>(string name)
        where T : unmanaged => this.DeclareBytesVar(name, (uint)sizeof(T));

    public ValueId DeclareBufferVar(string name, Access access)
    {
        var utf8 = Utf8Buffer.Encode(name, stackalloc byte[StackLimit]);
        fixed (byte* ptr = utf8)
            return VirException.CheckId(PInvoke.ModuleDeclareBufferVar(this.Handle, ptr, (nuint)utf8.Length, (ulong)access));
    }

    public ValueId DeclareU32Var(string name, uint value)
    {
        var utf8 = Utf8Buffer.Encode(name, stackalloc byte[StackLimit]);
        fixed (byte* ptr = utf8)
            return VirException.CheckId(PInvoke.ModuleDeclareU32Var(this.Handle, ptr, (nuint)utf8.Length, value));
    }

    public ValueId Clear(ValueId attachment, ClearValue color) =>
        VirException.CheckId(PInvoke.ModuleClear(this.Handle, attachment.Raw, color.ToNative()));

    public ValueId Present(ValueId attachment) => VirException.CheckId(PInvoke.ModulePresent(this.Handle, attachment.Raw));

    public Module BeginRendering(params ReadOnlySpan<(ValueId, Access)> attachments)
    {
        var native = attachments.Length <= StackLimit
            ? stackalloc VirAttachment[attachments.Length]
            : new VirAttachment[attachments.Length];
        for (var i = 0; i < attachments.Length; i++)
            native[i] = new VirAttachment { Value = attachments[i].Item1.Raw, Access = (ulong)attachments[i].Item2 };

        fixed (VirAttachment* ptr = native)
            PInvoke.ModuleBeginRendering(this.Handle, ptr, (nuint)native.Length);

        VirException.ThrowIfPending();
        return this;
    }

    public Module WithName(string name)
    {
        var utf8 = Utf8Buffer.Encode(name, stackalloc byte[StackLimit]);
        fixed (byte* ptr = utf8)
            PInvoke.ModuleWithName(this.Handle, ptr, (nuint)utf8.Length);

        return this.Checked();
    }

    public Module BindGraphicsPipeline(PipelineId pipeline)
    {
        PInvoke.ModuleBindGraphicsPipeline(this.Handle, pipeline.Raw);
        return this.Checked();
    }

    public Module SetViewport(uint index, Viewport viewport)
    {
        PInvoke.ModuleSetViewport(this.Handle, index, viewport.ToNative());
        return this.Checked();
    }

    public Module SetScissor(uint index, Rect2D rect)
    {
        PInvoke.ModuleSetScissor(this.Handle, index, rect.ToNative());
        return this.Checked();
    }

    public Module BroadcastColorBlend(BlendPreset preset)
    {
        PInvoke.ModuleBroadcastColorBlend(this.Handle, (uint)preset);
        return this.Checked();
    }

    public Module SetRasterization(RasterizationState state)
    {
        PInvoke.ModuleSetRasterization(this.Handle, state.ToNative());
        return this.Checked();
    }

    public Module SetDynamicState(DynamicStateFlags flags)
    {
        PInvoke.ModuleSetDynamicState(this.Handle, (uint)flags);
        return this.Checked();
    }

    public Module PushConstants<T>(in T value)
        where T : unmanaged => this.PushConstantsAt(0, in value);

    public Module PushConstantsAt<T>(uint offset, in T value)
        where T : unmanaged
    {
        fixed (T* ptr = &value)
            PInvoke.ModulePushConstantBytes(this.Handle, offset, (byte*)ptr, (nuint)sizeof(T));

        return this.Checked();
    }

    public Module PushConstantsFrom(ValueId variable) => this.PushConstantsFromAt(0, variable);

    public Module PushConstantsFromAt(uint offset, ValueId variable)
    {
        PInvoke.ModulePushConstantsFromAt(this.Handle, offset, variable.Raw);
        return this.Checked();
    }

    public Module Draw(uint vertexCount, uint instanceCount)
    {
        PInvoke.ModuleDraw(this.Handle, vertexCount, instanceCount);
        return this.Checked();
    }

    public void EndRendering(Span<ValueId> results)
    {
        fixed (ValueId* ptr = results)
            VirException.ThrowIfFailed(PInvoke.ModuleEndRendering(this.Handle, (uint*)ptr, (nuint)results.Length));
    }

    public void EndRendering(out ValueId a)
    {
        Span<ValueId> results = stackalloc ValueId[1];
        this.EndRendering(results);
        a = results[0];
    }

    public void EndRendering(out ValueId a, out ValueId b)
    {
        Span<ValueId> results = stackalloc ValueId[2];
        this.EndRendering(results);
        (a, b) = (results[0], results[1]);
    }

    public void EndRendering(out ValueId a, out ValueId b, out ValueId c)
    {
        Span<ValueId> results = stackalloc ValueId[3];
        this.EndRendering(results);
        (a, b, c) = (results[0], results[1], results[2]);
    }

    public void EndRendering(out ValueId a, out ValueId b, out ValueId c, out ValueId d)
    {
        Span<ValueId> results = stackalloc ValueId[4];
        this.EndRendering(results);
        (a, b, c, d) = (results[0], results[1], results[2], results[3]);
    }

    public Module BeginCompute(params ReadOnlySpan<(ValueId, Access)> attachments)
    {
        var native = attachments.Length <= StackLimit
            ? stackalloc VirAttachment[attachments.Length]
            : new VirAttachment[attachments.Length];
        for (var i = 0; i < attachments.Length; i++)
            native[i] = new VirAttachment { Value = attachments[i].Item1.Raw, Access = (ulong)attachments[i].Item2 };

        fixed (VirAttachment* ptr = native)
            PInvoke.ModuleBeginCompute(this.Handle, ptr, (nuint)native.Length);

        VirException.ThrowIfPending();
        return this;
    }

    public Module BindComputePipeline(PipelineId pipeline)
    {
        PInvoke.ModuleBindComputePipeline(this.Handle, pipeline.Raw);
        return this.Checked();
    }

    public Module Dispatch(uint x, uint y, uint z)
    {
        PInvoke.ModuleDispatch(this.Handle, x, y, z);
        return this.Checked();
    }

    public Module DispatchInvocations(uint x, uint y, uint z)
    {
        PInvoke.ModuleDispatchInvocations(this.Handle, x, y, z);
        return this.Checked();
    }

    public Module DispatchInvocationsPerPixel(ValueId image, float scaleX = 1, float scaleY = 1, float scaleZ = 1)
    {
        PInvoke.ModuleDispatchInvocationsPerPixelScaled(this.Handle, image.Raw, scaleX, scaleY, scaleZ);
        return this.Checked();
    }

    public Module DispatchInvocationsPerElement(ValueId buffer, ulong elementSize, float scale = 1)
    {
        PInvoke.ModuleDispatchInvocationsPerElementScaled(this.Handle, buffer.Raw, elementSize, scale);
        return this.Checked();
    }

    public Module DispatchIndirect(ValueId buffer, ulong offset = 0)
    {
        PInvoke.ModuleDispatchIndirectAt(this.Handle, buffer.Raw, offset);
        return this.Checked();
    }

    public Module BindImage(uint set, uint binding, ValueId image)
    {
        PInvoke.ModuleBindImage(this.Handle, set, binding, image.Raw);
        return this.Checked();
    }

    public Module BindBuffer(uint set, uint binding, ValueId buffer, ulong offset = 0, ulong range = ulong.MaxValue)
    {
        PInvoke.ModuleBindBufferRange(this.Handle, set, binding, buffer.Raw, offset, range);
        return this.Checked();
    }

    public Module BindSampler(uint set, uint binding, Sampler sampler)
    {
        PInvoke.ModuleBindSampler(this.Handle, set, binding, sampler.Raw);
        return this.Checked();
    }

    public Module PushConstantAddress(uint offset, ValueId buffer)
    {
        PInvoke.ModulePushConstantAddress(this.Handle, offset, buffer.Raw);
        return this.Checked();
    }

    public Module SpecializeConstant(uint id, bool value) => this.SpecializeConstant(id, 0, value ? 1UL : 0UL);

    public Module SpecializeConstant(uint id, uint value) => this.SpecializeConstant(id, 1, value);

    public Module SpecializeConstant(uint id, int value) => this.SpecializeConstant(id, 2, (uint)value);

    public Module SpecializeConstant(uint id, float value) => this.SpecializeConstant(id, 3, BitConverter.SingleToUInt32Bits(value));

    public Module SpecializeConstant(uint id, double value) => this.SpecializeConstant(id, 4, BitConverter.DoubleToUInt64Bits(value));

    private Module SpecializeConstant(uint id, uint kind, ulong bits)
    {
        PInvoke.ModuleSpecializeConstant(this.Handle, id, kind, bits);
        return this.Checked();
    }

    public ValueId TransientImage(in TransientImageInfo info)
    {
        var name = Utf8Buffer.Encode(info.Name ?? "", stackalloc byte[StackLimit]);
        fixed (byte* namePtr = name)
        {
            var native = new VirTransientImageInfo
            {
                Width = info.Width,
                Height = info.Height,
                Depth = info.Depth,
                Format = (int)info.Format,
                Usage = (uint)info.Usage,
                MipLevels = info.MipLevels,
                ArrayLayers = info.ArrayLayers,
                Samples = (uint)info.Samples,
                Name = namePtr,
                NameLength = (nuint)name.Length,
            };
            return VirException.CheckId(PInvoke.ModuleTransientImage(this.Handle, &native));
        }
    }

    public ValueId TransientBuffer(in BufferInfo info)
    {
        var name = Utf8Buffer.Encode(info.Name ?? "", stackalloc byte[StackLimit]);
        fixed (byte* namePtr = name)
        {
            var native = new VirBufferInfo
            {
                Size = info.Size,
                Usage = (uint)info.Usage,
                Location = (uint)info.Location,
                Alignment = info.Alignment,
                Name = namePtr,
                NameLength = (nuint)name.Length,
            };
            return VirException.CheckId(PInvoke.ModuleTransientBuffer(this.Handle, &native));
        }
    }

    public void EndCompute(Span<ValueId> results)
    {
        fixed (ValueId* ptr = results)
            VirException.ThrowIfFailed(PInvoke.ModuleEndCompute(this.Handle, (uint*)ptr, (nuint)results.Length));
    }

    public void EndCompute(out ValueId a)
    {
        Span<ValueId> results = stackalloc ValueId[1];
        this.EndCompute(results);
        a = results[0];
    }

    public void EndCompute(out ValueId a, out ValueId b)
    {
        Span<ValueId> results = stackalloc ValueId[2];
        this.EndCompute(results);
        (a, b) = (results[0], results[1]);
    }

    public void EndCompute(out ValueId a, out ValueId b, out ValueId c)
    {
        Span<ValueId> results = stackalloc ValueId[3];
        this.EndCompute(results);
        (a, b, c) = (results[0], results[1], results[2]);
    }

    public ValueId SetCondition(ValueId condition, Func<Module, ValueId> then, Func<Module, ValueId> otherwise)
    {
        Span<ValueId> results = stackalloc ValueId[1];
        this.SetCondition(condition, m => ValueTuple.Create(then(m)), m => ValueTuple.Create(otherwise(m)), results);
        return results[0];
    }

    public (ValueId, ValueId) SetCondition(
        ValueId condition,
        Func<Module, (ValueId, ValueId)> then,
        Func<Module, (ValueId, ValueId)> otherwise
    )
    {
        Span<ValueId> results = stackalloc ValueId[2];
        this.SetCondition(condition, m => then(m), m => otherwise(m), results);
        return (results[0], results[1]);
    }

    public (ValueId, ValueId, ValueId) SetCondition(
        ValueId condition,
        Func<Module, (ValueId, ValueId, ValueId)> then,
        Func<Module, (ValueId, ValueId, ValueId)> otherwise
    )
    {
        Span<ValueId> results = stackalloc ValueId[3];
        this.SetCondition(condition, m => then(m), m => otherwise(m), results);
        return (results[0], results[1], results[2]);
    }

    public (ValueId, ValueId, ValueId, ValueId) SetCondition(
        ValueId condition,
        Func<Module, (ValueId, ValueId, ValueId, ValueId)> then,
        Func<Module, (ValueId, ValueId, ValueId, ValueId)> otherwise
    )
    {
        Span<ValueId> results = stackalloc ValueId[4];
        this.SetCondition(condition, m => then(m), m => otherwise(m), results);
        return (results[0], results[1], results[2], results[3]);
    }

    private void SetCondition(ValueId condition, Func<Module, ITuple> then, Func<Module, ITuple> otherwise, Span<ValueId> results)
    {
        var state = new ConditionState { Then = then, Otherwise = otherwise };
        var gcHandle = GCHandle.Alloc(state);
        try
        {
            nuint count;
            VkResult result;
            fixed (ValueId* ptr = results)
            {
                result = PInvoke.ModuleSetCondition(
                    this.Handle,
                    condition.Raw,
                    &ThenThunk,
                    &OtherwiseThunk,
                    GCHandle.ToIntPtr(gcHandle),
                    (uint*)ptr,
                    (nuint)results.Length,
                    &count
                );
            }

            if (state.Error is not null)
            {
                VirException.TakeLastError();
                state.Error.Throw();
            }

            VirException.ThrowIfFailed(result);
        }
        finally
        {
            gcHandle.Free();
        }
    }

    public ValueId AcquireNextImage(SwapChain swapchain) =>
        VirException.CheckId(PInvoke.ModuleAcquireNextImage(this.Handle, swapchain.Handle));

    public Program Compile(RenderGraph graph, ValueId root)
    {
        nint program;
        VirException.ThrowIfFailed(PInvoke.ModuleCompile(this.Handle, graph.Handle, root.Raw, &program));
        return new Program(program);
    }

    public Program CompileAll(RenderGraph graph, ReadOnlySpan<ValueId> roots)
    {
        var raw = roots.Length <= StackLimit ? stackalloc uint[roots.Length] : new uint[roots.Length];
        for (var i = 0; i < roots.Length; i++)
            raw[i] = roots[i].Raw;

        nint program;
        fixed (uint* ptr = raw)
            VirException.ThrowIfFailed(PInvoke.ModuleCompileAll(this.Handle, graph.Handle, ptr, (nuint)raw.Length, &program));

        return new Program(program);
    }

    public ValueId ImportBuffer(GpuBuffer buffer, Access access) =>
        VirException.CheckId(PInvoke.ModuleImportBuffer(this.Handle, buffer.Handle, (ulong)access));

    public ValueId ImportImage(GpuImage image, ImageLayout layout, Access access) =>
        VirException.CheckId(PInvoke.ModuleImportImage(this.Handle, image.Handle, (int)layout, (ulong)access));

    public ValueId SetName(ValueId value, string name)
    {
        var utf8 = Utf8Buffer.Encode(name, stackalloc byte[StackLimit]);
        fixed (byte* ptr = utf8)
            return VirException.CheckId(PInvoke.ModuleSetName(this.Handle, value.Raw, ptr, (nuint)utf8.Length));
    }

    public ValueId CopyBufferToImage(ValueId buffer, ValueId image) =>
        VirException.CheckId(PInvoke.ModuleCopyBufferToImage(this.Handle, buffer.Raw, image.Raw));

    public ValueId Export(ValueId value, Access access, DomainFlag domain) =>
        VirException.CheckId(PInvoke.ModuleExport(this.Handle, value.Raw, (ulong)access, (uint)domain));

    public Module SetDepth(DepthState depth)
    {
        PInvoke.ModuleSetDepth(this.Handle, depth.ToNative());
        return this.Checked();
    }

    public Module BindTexture(uint set, uint binding, ValueId image, Sampler sampler)
    {
        PInvoke.ModuleBindTexture(this.Handle, set, binding, image.Raw, sampler.Raw);
        return this.Checked();
    }

    public Module BindVertexBuffer(uint binding, ValueId buffer)
    {
        PInvoke.ModuleBindVertexBuffer(this.Handle, binding, buffer.Raw);
        return this.Checked();
    }

    public Module BindIndexBuffer(ValueId buffer, IndexType indexType)
    {
        PInvoke.ModuleBindIndexBuffer(this.Handle, buffer.Raw, (int)indexType);
        return this.Checked();
    }

    public Module DrawIndexed(uint indexCount, uint instanceCount)
    {
        PInvoke.ModuleDrawIndexed(this.Handle, indexCount, instanceCount);
        return this.Checked();
    }

    public Module BindIndexBuffer(ValueId buffer, ulong offset, IndexType indexType)
    {
        PInvoke.ModuleBindIndexBufferAt(this.Handle, buffer.Raw, offset, (int)indexType);
        return this.Checked();
    }

    public Module DrawRange(uint vertexCount, uint instanceCount, uint firstVertex = 0, uint firstInstance = 0)
    {
        PInvoke.ModuleDrawRange(this.Handle, vertexCount, instanceCount, firstVertex, firstInstance);
        return this.Checked();
    }

    public Module DrawIndexedRange(
        uint indexCount,
        uint instanceCount,
        uint firstIndex = 0,
        int vertexOffset = 0,
        uint firstInstance = 0
    )
    {
        PInvoke.ModuleDrawIndexedRange(this.Handle, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        return this.Checked();
    }

    public Module DrawIndirect(ValueId buffer, uint drawCount, ulong offset = 0, uint stride = 16)
    {
        PInvoke.ModuleDrawIndirectAt(this.Handle, buffer.Raw, offset, drawCount, stride);
        return this.Checked();
    }

    public Module DrawIndexedIndirect(ValueId buffer, uint drawCount, ulong offset = 0, uint stride = 20)
    {
        PInvoke.ModuleDrawIndexedIndirectAt(this.Handle, buffer.Raw, offset, drawCount, stride);
        return this.Checked();
    }

    public Module DrawIndirectCount(
        ValueId buffer,
        ValueId countBuffer,
        uint maxDrawCount,
        ulong offset = 0,
        ulong countOffset = 0,
        uint stride = 16
    )
    {
        PInvoke.ModuleDrawIndirectCountAt(this.Handle, buffer.Raw, offset, countBuffer.Raw, countOffset, maxDrawCount, stride);
        return this.Checked();
    }

    public Module DrawIndexedIndirectCount(
        ValueId buffer,
        ValueId countBuffer,
        uint maxDrawCount,
        ulong offset = 0,
        ulong countOffset = 0,
        uint stride = 20
    )
    {
        PInvoke.ModuleDrawIndexedIndirectCountAt(this.Handle, buffer.Raw, offset, countBuffer.Raw, countOffset, maxDrawCount, stride);
        return this.Checked();
    }

    public ValueId CopyBuffer(ValueId src, ValueId dst, params ReadOnlySpan<BufferCopy> regions)
    {
        Span<VirBufferCopy> native = regions.Length <= StackLimit / sizeof(VirBufferCopy)
            ? stackalloc VirBufferCopy[regions.Length]
            : new VirBufferCopy[regions.Length];
        for (var i = 0; i < regions.Length; i++)
            native[i] = new VirBufferCopy
            {
                SrcOffset = regions[i].SrcOffset,
                DstOffset = regions[i].DstOffset,
                Size = regions[i].Size,
            };

        fixed (VirBufferCopy* ptr = native)
            return VirException.CheckId(PInvoke.ModuleCopyBuffer(this.Handle, src.Raw, dst.Raw, ptr, (nuint)native.Length));
    }

    public ValueId CopyBufferToImage(ValueId buffer, ValueId image, params ReadOnlySpan<BufferImageCopy> regions)
    {
        var native = new VirBufferImageCopy[regions.Length];
        for (var i = 0; i < regions.Length; i++)
            native[i] = new VirBufferImageCopy
            {
                BufferOffset = regions[i].BufferOffset,
                MipLevel = regions[i].MipLevel,
                OffsetX = regions[i].OffsetX,
                OffsetY = regions[i].OffsetY,
                OffsetZ = regions[i].OffsetZ,
                Width = regions[i].Width,
                Height = regions[i].Height,
                Depth = regions[i].Depth,
            };

        fixed (VirBufferImageCopy* ptr = native)
            return VirException.CheckId(
                PInvoke.ModuleCopyBufferToImageRegion(this.Handle, buffer.Raw, image.Raw, ptr, (nuint)native.Length)
            );
    }
    private sealed class ConditionState
    {
        public required Func<Module, ITuple> Then;
        public required Func<Module, ITuple> Otherwise;
        public ExceptionDispatchInfo? Error;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static nuint ThenThunk(nint module, nint userData, uint* output, nuint capacity) =>
        RunBranch(module, userData, output, capacity, isThen: true);

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static nuint OtherwiseThunk(nint module, nint userData, uint* output, nuint capacity) =>
        RunBranch(module, userData, output, capacity, isThen: false);

    private static nuint RunBranch(nint module, nint userData, uint* output, nuint capacity, bool isThen)
    {
        var state = (ConditionState)GCHandle.FromIntPtr(userData).Target!;
        try
        {
            var branch = isThen ? state.Then : state.Otherwise;
            var results = branch(new Module(module));
            for (var i = 0; i < results.Length; i++)
                output[i] = ((ValueId)results[i]!).Raw;

            return (nuint)results.Length;
        }
        catch (Exception exception)
        {
            state.Error ??= ExceptionDispatchInfo.Capture(exception);
            return 0;
        }
    }

    private Module Checked()
    {
        VirException.ThrowIfPending();
        return this;
    }
}
