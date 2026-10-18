using System.Text;

namespace Vir;

public sealed class VirException(VkResult result, string message) : Exception(message)
{
    public VkResult Result { get; } = result;

    internal static unsafe string TakeLastError()
    {
        var length = (int)PInvoke.LastError(null, 0);
        if (length == 0)
            return string.Empty;

        var buffer = length <= 512 ? stackalloc byte[length] : new byte[length];
        fixed (byte* ptr = buffer)
            PInvoke.LastError(ptr, (nuint)length);

        PInvoke.ClearLastError();
        return Encoding.UTF8.GetString(buffer);
    }

    internal static void ThrowIfFailed(VkResult result)
    {
        if (result != VkResult.Success)
            throw new VirException(result, TakeLastError());
    }

    internal static ValueId CheckId(uint id)
    {
        if (id == uint.MaxValue)
            throw new VirException(VkResult.ErrorUnknown, TakeLastError());

        return new ValueId(id);
    }

    internal static unsafe void ThrowIfPending()
    {
        if (PInvoke.LastError(null, 0) != 0)
            throw new VirException(VkResult.ErrorUnknown, TakeLastError());
    }
}
