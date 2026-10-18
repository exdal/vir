namespace Vir;

public sealed unsafe class Context : IDisposable
{
    private nint handle;

    private Context(nint handle) => this.handle = handle;

    public struct CreateInfo
    {
        public nint GetInstanceProcAddr;
        public nint Instance;
        public nint PhysicalDevice;
        public nint Device;
    }

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(Context));

    public static Context Create(in CreateInfo info)
    {
        var native = new VirContextCreateInfo
        {
            GetInstanceProcAddr = info.GetInstanceProcAddr,
            Instance = info.Instance,
            PhysicalDevice = info.PhysicalDevice,
            Device = info.Device,
        };

        nint handle;
        VirException.ThrowIfFailed(PInvoke.ContextCreate(&native, &handle));
        return new Context(handle);
    }

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        PInvoke.ContextDestroy(this.handle);
        this.handle = 0;
    }

    public void CreateCommandQueue(uint queueFamilyIndex, DomainFlag domain) =>
        VirException.ThrowIfFailed(PInvoke.ContextCreateCommandQueue(this.Handle, queueFamilyIndex, (uint)domain));

    public void WaitIdle() => VirException.ThrowIfFailed(PInvoke.ContextWaitIdle(this.Handle));

    public PersistentAllocator CreatePersistentAllocator()
    {
        nint allocator;
        VirException.ThrowIfFailed(PInvoke.PersistentAllocatorCreate(this.Handle, &allocator));
        return new PersistentAllocator(allocator);
    }

    public SuperFrameAllocator CreateSuperFrameAllocator(uint frameCount)
    {
        nint allocator;
        VirException.ThrowIfFailed(PInvoke.SuperFrameAllocatorCreate(this.Handle, frameCount, &allocator));
        return new SuperFrameAllocator(allocator);
    }
}
