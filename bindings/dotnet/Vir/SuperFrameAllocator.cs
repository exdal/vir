using System.Text;

namespace Vir;

public readonly record struct PassTiming(string Name, TimeSpan Duration);

public readonly unsafe struct FrameAllocator
{
    internal FrameAllocator(nint handle) => this.Handle = handle;

    internal nint Handle { get; }

    public PassTiming[] GetCompletedTimings()
    {
        nuint count;
        VirException.ThrowIfFailed(PInvoke.FrameAllocatorCompletedTimings(this.Handle, null, 0, &count));
        if (count == 0)
            return [];

        var native = new VirPassTiming[count];
        fixed (VirPassTiming* ptr = native)
        {
            VirException.ThrowIfFailed(PInvoke.FrameAllocatorCompletedTimings(this.Handle, ptr, (nuint)native.Length, &count));
            var timings = new PassTiming[Math.Min((int)count, native.Length)];
            for (var i = 0; i < timings.Length; i++)
            {
                var name = Encoding.UTF8.GetString(ptr[i].Name, (int)ptr[i].NameLength);
                timings[i] = new PassTiming(name, TimeSpan.FromTicks((long)(ptr[i].DurationNs / 100)));
            }

            return timings;
        }
    }

    public void AddTimelineWait(ulong semaphore, ulong value) =>
        PInvoke.FrameAllocatorAddTimelineWait(this.Handle, semaphore, value);
}

public sealed unsafe class SuperFrameAllocator : IDisposable
{
    private nint handle;

    internal SuperFrameAllocator(nint handle) => this.handle = handle;

    private nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(SuperFrameAllocator));

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        PInvoke.SuperFrameAllocatorDestroy(this.handle);
        this.handle = 0;
    }

    public bool EnableTimestamps(Context context)
    {
        uint enabled;
        VirException.ThrowIfFailed(PInvoke.SuperFrameAllocatorEnableTimestamps(this.Handle, context.Handle, &enabled));
        return enabled != 0;
    }

    public FrameAllocator GetNextFrame()
    {
        nint frame;
        VirException.ThrowIfFailed(PInvoke.SuperFrameAllocatorNextFrame(this.Handle, &frame));
        return new FrameAllocator(frame);
    }
}
