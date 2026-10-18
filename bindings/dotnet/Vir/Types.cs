using System.Runtime.InteropServices;

namespace Vir;

public readonly record struct ValueId(uint Raw)
{
    public static ValueId Invalid => new(uint.MaxValue);

    public bool IsValid => this.Raw != uint.MaxValue;
}

public readonly record struct PipelineId(uint Raw)
{
    public static PipelineId Invalid => new(uint.MaxValue);
}

[Flags]
public enum DynamicStateFlags : uint
{
    None = 0,
    Viewport = 1 << 0,
    Scissor = 1 << 1,
}

public enum BlendPreset : uint
{
    Off,
    AlphaBlend,
    PremultipliedAlphaBlend,
    Additive,
}

public enum PolygonMode
{
    Fill = 0,
    Line = 1,
    Point = 2,
}

[Flags]
public enum CullMode : uint
{
    None = 0,
    Front = 1,
    Back = 2,
    FrontAndBack = 3,
}

public enum FrontFace
{
    CounterClockwise = 0,
    Clockwise = 1,
}

public struct RasterizationState()
{
    public PolygonMode PolygonMode = PolygonMode.Fill;
    public CullMode CullMode = CullMode.None;
    public FrontFace FrontFace = FrontFace.CounterClockwise;
    public float LineWidth = 1.0f;
    public bool DepthClampEnable = false;
    public bool RasterizerDiscardEnable = false;

    internal readonly VirRasterizationState ToNative() =>
        new()
        {
            PolygonMode = (int)this.PolygonMode,
            CullMode = (uint)this.CullMode,
            FrontFace = (int)this.FrontFace,
            LineWidth = this.LineWidth,
            DepthClampEnable = this.DepthClampEnable ? 1u : 0u,
            RasterizerDiscardEnable = this.RasterizerDiscardEnable ? 1u : 0u,
        };
}

public readonly struct Rect2D
{
    private readonly VirRect2D native;

    private Rect2D(VirRect2D native) => this.native = native;

    public static Rect2D Framebuffer() => new(new VirRect2D { Tag = 0 });

    public static Rect2D Absolute(int x, int y, uint width, uint height) =>
        new(
            new VirRect2D
            {
                Tag = 1,
                X = x,
                Y = y,
                Width = width,
                Height = height,
            }
        );

    public static Rect2D Relative(float x, float y, float width, float height) =>
        new(
            new VirRect2D
            {
                Tag = 2,
                RelativeX = x,
                RelativeY = y,
                RelativeWidth = width,
                RelativeHeight = height,
            }
        );

    internal VirRect2D ToNative() => this.native;
}

public readonly record struct Viewport(Rect2D Rect, float MinDepth = 0.0f, float MaxDepth = 1.0f)
{
    public static implicit operator Viewport(Rect2D rect) => new(rect);

    internal VirViewport ToNative() =>
        new()
        {
            Rect = this.Rect.ToNative(),
            MinDepth = this.MinDepth,
            MaxDepth = this.MaxDepth,
        };
}

public readonly struct ClearValue
{
    private readonly VirClearValue native;

    private ClearValue(VirClearValue native) => this.native = native;

    public static ClearValue Rgba(float r, float g, float b, float a)
    {
        Span<float> color = [r, g, b, a];
        return new(MemoryMarshal.Read<VirClearValue>(MemoryMarshal.AsBytes(color)));
    }

    public static ClearValue Depth(float depth, uint stencil = 0)
    {
        Span<uint> value = [BitConverter.SingleToUInt32Bits(depth), stencil, 0, 0];
        return new(MemoryMarshal.Read<VirClearValue>(MemoryMarshal.AsBytes(value)));
    }

    internal VirClearValue ToNative() => this.native;
}
