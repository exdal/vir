using System.Text;

namespace Vir;

internal static class Utf8Buffer
{
    public const int StackLimit = 256;

    public static Span<byte> Encode(string text, Span<byte> scratch)
    {
        var length = Encoding.UTF8.GetByteCount(text);
        var buffer = length <= scratch.Length ? scratch[..length] : new byte[length];
        Encoding.UTF8.GetBytes(text, buffer);
        return buffer;
    }
}
