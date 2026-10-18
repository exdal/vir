namespace Vir;

public sealed unsafe class SwapChain : IDisposable
{
    private readonly Context context;
    private nint handle;

    public SwapChain(
        Context context,
        PersistentAllocator allocator,
        ulong swapchain,
        ReadOnlySpan<ulong> images,
        int format,
        uint width,
        uint height
    )
    {
        this.context = context;
        nint handle;
        fixed (ulong* imagesPtr = images)
        {
            var native = new VirSwapChainInfo
            {
                SwapChain = swapchain,
                Images = imagesPtr,
                ImageCount = (uint)images.Length,
                Format = format,
                Width = width,
                Height = height,
            };
            VirException.ThrowIfFailed(PInvoke.SwapChainCreate(allocator.Handle, &native, &handle));
        }

        this.handle = handle;
    }

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(SwapChain));

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        PInvoke.SwapChainDestroy(this.context.Handle, this.handle);
        this.handle = 0;
    }
}
