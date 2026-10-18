namespace Vir;

public sealed unsafe class PersistentAllocator : IDisposable
{
    private nint handle;

    internal PersistentAllocator(nint handle) => this.handle = handle;

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(PersistentAllocator));

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        PInvoke.PersistentAllocatorDestroy(this.handle);
        this.handle = 0;
    }

    public GpuBuffer AllocateBuffer(in BufferInfo info)
    {
        var name = Utf8Buffer.Encode(info.Name ?? "", stackalloc byte[Utf8Buffer.StackLimit]);
        nint buffer;
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
            VirException.ThrowIfFailed(PInvoke.PersistentAllocatorAllocateBuffer(this.Handle, &native, &buffer));
        }

        return new GpuBuffer(buffer);
    }

    public void DeallocateBuffer(GpuBuffer buffer) => PInvoke.PersistentAllocatorDeallocateBuffer(this.Handle, buffer.Release());

    public GpuImage AllocateImage(in ImageInfo info)
    {
        var name = Utf8Buffer.Encode(info.Name ?? "", stackalloc byte[Utf8Buffer.StackLimit]);
        nint image;
        fixed (byte* namePtr = name)
        {
            var native = new VirImageInfo
            {
                Kind = info.Kind,
                Width = info.Width,
                Height = info.Height,
                Format = (int)info.Format,
                Usage = (uint)info.Usage,
                MipLevels = info.MipLevels,
                ArrayLayers = info.ArrayLayers,
                Samples = (uint)info.Samples,
                Name = namePtr,
                NameLength = (nuint)name.Length,
            };
            VirException.ThrowIfFailed(PInvoke.PersistentAllocatorAllocateImage(this.Handle, &native, &image));
        }

        return new GpuImage(image);
    }

    public void DeallocateImage(GpuImage image) => PInvoke.PersistentAllocatorDeallocateImage(this.Handle, image.Release());

    public Sampler AllocateSampler(in SamplerInfo info)
    {
        var native = info.ToNative();
        ulong sampler;
        VirException.ThrowIfFailed(PInvoke.PersistentAllocatorAllocateSampler(this.Handle, &native, &sampler));
        return new Sampler(sampler);
    }

    public void DeallocateSampler(Sampler sampler) => PInvoke.PersistentAllocatorDeallocateSampler(this.Handle, sampler.Raw);
}
