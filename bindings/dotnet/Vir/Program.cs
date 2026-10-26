namespace Vir;

public sealed unsafe class Program : IDisposable
{
    private nint handle;

    internal Program(nint handle) => this.handle = handle;

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(Program));

    public void Dispose()
    {
        if (this.handle == 0)
            return;

        PInvoke.ProgramDestroy(this.handle);
        this.handle = 0;
    }

    public void Set(ValueId variable, bool value) =>
        VirException.ThrowIfFailed(PInvoke.ProgramSetBool(this.Handle, variable.Raw, value ? (byte)1 : (byte)0));

    public void Set(ValueId variable, uint value) =>
        VirException.ThrowIfFailed(PInvoke.ProgramSetU32(this.Handle, variable.Raw, value));

    public void Set(ValueId variable, GpuBuffer buffer) =>
        VirException.ThrowIfFailed(PInvoke.ProgramSetBuffer(this.Handle, variable.Raw, buffer.Handle));

    public void SetBytes<T>(ValueId variable, in T value)
        where T : unmanaged
    {
        fixed (T* ptr = &value)
            VirException.ThrowIfFailed(PInvoke.ProgramSetBytes(this.Handle, variable.Raw, (byte*)ptr, (nuint)sizeof(T)));
    }
}
