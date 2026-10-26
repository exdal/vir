using System.Runtime.InteropServices;

namespace Vir;

public enum Format
{
    Undefined = 0,
    R4G4UnormPack8 = 1,
    R4G4B4A4UnormPack16 = 2,
    B4G4R4A4UnormPack16 = 3,
    R5G6B5UnormPack16 = 4,
    B5G6R5UnormPack16 = 5,
    R5G5B5A1UnormPack16 = 6,
    B5G5R5A1UnormPack16 = 7,
    A1R5G5B5UnormPack16 = 8,
    R8Unorm = 9,
    R8Snorm = 10,
    R8Uscaled = 11,
    R8Sscaled = 12,
    R8Uint = 13,
    R8Sint = 14,
    R8Srgb = 15,
    R8G8Unorm = 16,
    R8G8Snorm = 17,
    R8G8Uscaled = 18,
    R8G8Sscaled = 19,
    R8G8Uint = 20,
    R8G8Sint = 21,
    R8G8Srgb = 22,
    R8G8B8Unorm = 23,
    R8G8B8Snorm = 24,
    R8G8B8Uscaled = 25,
    R8G8B8Sscaled = 26,
    R8G8B8Uint = 27,
    R8G8B8Sint = 28,
    R8G8B8Srgb = 29,
    B8G8R8Unorm = 30,
    B8G8R8Snorm = 31,
    B8G8R8Uscaled = 32,
    B8G8R8Sscaled = 33,
    B8G8R8Uint = 34,
    B8G8R8Sint = 35,
    B8G8R8Srgb = 36,
    R8G8B8A8Unorm = 37,
    R8G8B8A8Snorm = 38,
    R8G8B8A8Uscaled = 39,
    R8G8B8A8Sscaled = 40,
    R8G8B8A8Uint = 41,
    R8G8B8A8Sint = 42,
    R8G8B8A8Srgb = 43,
    B8G8R8A8Unorm = 44,
    B8G8R8A8Snorm = 45,
    B8G8R8A8Uscaled = 46,
    B8G8R8A8Sscaled = 47,
    B8G8R8A8Uint = 48,
    B8G8R8A8Sint = 49,
    B8G8R8A8Srgb = 50,
    A8B8G8R8UnormPack32 = 51,
    A8B8G8R8SnormPack32 = 52,
    A8B8G8R8UscaledPack32 = 53,
    A8B8G8R8SscaledPack32 = 54,
    A8B8G8R8UintPack32 = 55,
    A8B8G8R8SintPack32 = 56,
    A8B8G8R8SrgbPack32 = 57,
    A2R10G10B10UnormPack32 = 58,
    A2R10G10B10SnormPack32 = 59,
    A2R10G10B10UscaledPack32 = 60,
    A2R10G10B10SscaledPack32 = 61,
    A2R10G10B10UintPack32 = 62,
    A2R10G10B10SintPack32 = 63,
    A2B10G10R10UnormPack32 = 64,
    A2B10G10R10SnormPack32 = 65,
    A2B10G10R10UscaledPack32 = 66,
    A2B10G10R10SscaledPack32 = 67,
    A2B10G10R10UintPack32 = 68,
    A2B10G10R10SintPack32 = 69,
    R16Unorm = 70,
    R16Snorm = 71,
    R16Uscaled = 72,
    R16Sscaled = 73,
    R16Uint = 74,
    R16Sint = 75,
    R16Sfloat = 76,
    R16G16Unorm = 77,
    R16G16Snorm = 78,
    R16G16Uscaled = 79,
    R16G16Sscaled = 80,
    R16G16Uint = 81,
    R16G16Sint = 82,
    R16G16Sfloat = 83,
    R16G16B16Unorm = 84,
    R16G16B16Snorm = 85,
    R16G16B16Uscaled = 86,
    R16G16B16Sscaled = 87,
    R16G16B16Uint = 88,
    R16G16B16Sint = 89,
    R16G16B16Sfloat = 90,
    R16G16B16A16Unorm = 91,
    R16G16B16A16Snorm = 92,
    R16G16B16A16Uscaled = 93,
    R16G16B16A16Sscaled = 94,
    R16G16B16A16Uint = 95,
    R16G16B16A16Sint = 96,
    R16G16B16A16Sfloat = 97,
    R32Uint = 98,
    R32Sint = 99,
    R32Sfloat = 100,
    R32G32Uint = 101,
    R32G32Sint = 102,
    R32G32Sfloat = 103,
    R32G32B32Uint = 104,
    R32G32B32Sint = 105,
    R32G32B32Sfloat = 106,
    R32G32B32A32Uint = 107,
    R32G32B32A32Sint = 108,
    R32G32B32A32Sfloat = 109,
    R64Uint = 110,
    R64Sint = 111,
    R64Sfloat = 112,
    R64G64Uint = 113,
    R64G64Sint = 114,
    R64G64Sfloat = 115,
    R64G64B64Uint = 116,
    R64G64B64Sint = 117,
    R64G64B64Sfloat = 118,
    R64G64B64A64Uint = 119,
    R64G64B64A64Sint = 120,
    R64G64B64A64Sfloat = 121,
    B10G11R11UfloatPack32 = 122,
    E5B9G9R9UfloatPack32 = 123,
    D16Unorm = 124,
    X8D24UnormPack32 = 125,
    D32Sfloat = 126,
    S8Uint = 127,
    D16UnormS8Uint = 128,
    D24UnormS8Uint = 129,
    D32SfloatS8Uint = 130,
    Bc1RgbUnormBlock = 131,
    Bc1RgbSrgbBlock = 132,
    Bc1RgbaUnormBlock = 133,
    Bc1RgbaSrgbBlock = 134,
    Bc2UnormBlock = 135,
    Bc2SrgbBlock = 136,
    Bc3UnormBlock = 137,
    Bc3SrgbBlock = 138,
    Bc4UnormBlock = 139,
    Bc4SnormBlock = 140,
    Bc5UnormBlock = 141,
    Bc5SnormBlock = 142,
    Bc6hUfloatBlock = 143,
    Bc6hSfloatBlock = 144,
    Bc7UnormBlock = 145,
    Bc7SrgbBlock = 146,
    Etc2R8G8B8UnormBlock = 147,
    Etc2R8G8B8SrgbBlock = 148,
    Etc2R8G8B8A1UnormBlock = 149,
    Etc2R8G8B8A1SrgbBlock = 150,
    Etc2R8G8B8A8UnormBlock = 151,
    Etc2R8G8B8A8SrgbBlock = 152,
    EacR11UnormBlock = 153,
    EacR11SnormBlock = 154,
    EacR11G11UnormBlock = 155,
    EacR11G11SnormBlock = 156,
    Astc4x4UnormBlock = 157,
    Astc4x4SrgbBlock = 158,
    Astc5x4UnormBlock = 159,
    Astc5x4SrgbBlock = 160,
    Astc5x5UnormBlock = 161,
    Astc5x5SrgbBlock = 162,
    Astc6x5UnormBlock = 163,
    Astc6x5SrgbBlock = 164,
    Astc6x6UnormBlock = 165,
    Astc6x6SrgbBlock = 166,
    Astc8x5UnormBlock = 167,
    Astc8x5SrgbBlock = 168,
    Astc8x6UnormBlock = 169,
    Astc8x6SrgbBlock = 170,
    Astc8x8UnormBlock = 171,
    Astc8x8SrgbBlock = 172,
    Astc10x5UnormBlock = 173,
    Astc10x5SrgbBlock = 174,
    Astc10x6UnormBlock = 175,
    Astc10x6SrgbBlock = 176,
    Astc10x8UnormBlock = 177,
    Astc10x8SrgbBlock = 178,
    Astc10x10UnormBlock = 179,
    Astc10x10SrgbBlock = 180,
    Astc12x10UnormBlock = 181,
    Astc12x10SrgbBlock = 182,
    Astc12x12UnormBlock = 183,
    Astc12x12SrgbBlock = 184,
    Pvrtc12BppUnormBlock = 1000054000,
    Pvrtc14BppUnormBlock = 1000054001,
    Pvrtc22BppUnormBlock = 1000054002,
    Pvrtc24BppUnormBlock = 1000054003,
    Pvrtc12BppSrgbBlock = 1000054004,
    Pvrtc14BppSrgbBlock = 1000054005,
    Pvrtc22BppSrgbBlock = 1000054006,
    Pvrtc24BppSrgbBlock = 1000054007,
    R16G16S105 = 1000464000,
    A1B5G5R5UnormPack16 = 1000470000,
    A8Unorm = 1000470001,
    G8B8G8R8422Unorm = 1000156000,
    B8G8R8G8422Unorm = 1000156001,
    G8B8R83Plane420Unorm = 1000156002,
    G8B8R82Plane420Unorm = 1000156003,
    G8B8R83Plane422Unorm = 1000156004,
    G8B8R82Plane422Unorm = 1000156005,
    G8B8R83Plane444Unorm = 1000156006,
    R10X6UnormPack16 = 1000156007,
    R10X6G10X6Unorm2Pack16 = 1000156008,
    R10X6G10X6B10X6A10X6Unorm4Pack16 = 1000156009,
    G10X6B10X6G10X6R10X6422Unorm4Pack16 = 1000156010,
    B10X6G10X6R10X6G10X6422Unorm4Pack16 = 1000156011,
    G10X6B10X6R10X63Plane420Unorm3Pack16 = 1000156012,
    G10X6B10X6R10X62Plane420Unorm3Pack16 = 1000156013,
    G10X6B10X6R10X63Plane422Unorm3Pack16 = 1000156014,
    G10X6B10X6R10X62Plane422Unorm3Pack16 = 1000156015,
    G10X6B10X6R10X63Plane444Unorm3Pack16 = 1000156016,
    R12X4UnormPack16 = 1000156017,
    R12X4G12X4Unorm2Pack16 = 1000156018,
    R12X4G12X4B12X4A12X4Unorm4Pack16 = 1000156019,
    G12X4B12X4G12X4R12X4422Unorm4Pack16 = 1000156020,
    B12X4G12X4R12X4G12X4422Unorm4Pack16 = 1000156021,
    G12X4B12X4R12X43Plane420Unorm3Pack16 = 1000156022,
    G12X4B12X4R12X42Plane420Unorm3Pack16 = 1000156023,
    G12X4B12X4R12X43Plane422Unorm3Pack16 = 1000156024,
    G12X4B12X4R12X42Plane422Unorm3Pack16 = 1000156025,
    G12X4B12X4R12X43Plane444Unorm3Pack16 = 1000156026,
    G16B16G16R16422Unorm = 1000156027,
    B16G16R16G16422Unorm = 1000156028,
    G16B16R163Plane420Unorm = 1000156029,
    G16B16R162Plane420Unorm = 1000156030,
    G16B16R163Plane422Unorm = 1000156031,
    G16B16R162Plane422Unorm = 1000156032,
    G16B16R163Plane444Unorm = 1000156033,
    G8B8R82Plane444Unorm = 1000330000,
    G10X6B10X6R10X62Plane444Unorm3Pack16 = 1000330001,
    G12X4B12X4R12X42Plane444Unorm3Pack16 = 1000330002,
    G16B16R162Plane444Unorm = 1000330003,
    A4R4G4B4UnormPack16 = 1000340000,
    A4B4G4R4UnormPack16 = 1000340001,
    Astc4x4SfloatBlock = 1000066000,
    Astc5x4SfloatBlock = 1000066001,
    Astc5x5SfloatBlock = 1000066002,
    Astc6x5SfloatBlock = 1000066003,
    Astc6x6SfloatBlock = 1000066004,
    Astc8x5SfloatBlock = 1000066005,
    Astc8x6SfloatBlock = 1000066006,
    Astc8x8SfloatBlock = 1000066007,
    Astc10x5SfloatBlock = 1000066008,
    Astc10x6SfloatBlock = 1000066009,
    Astc10x8SfloatBlock = 1000066010,
    Astc10x10SfloatBlock = 1000066011,
    Astc12x10SfloatBlock = 1000066012,
    Astc12x12SfloatBlock = 1000066013,
}

public enum Filter
{
    Nearest = 0,
    Linear = 1,
    Cubic = 1000015000,
}

public enum MipmapMode
{
    Nearest = 0,
    Linear = 1,
}

public enum SamplerAddressMode
{
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
    ClampToBorder = 3,
    MirrorClampToEdge = 4,
}

public enum CompareOp
{
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
}

public enum IndexType
{
    Uint16 = 0,
    Uint32 = 1,
    Uint8 = 1000265000,
}

public enum ImageLayout
{
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    DepthStencilReadOnlyOptimal = 4,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,
    Preinitialized = 8,
    PresentSrc = 1000001002,
    VideoDecodeDst = 1000024000,
    VideoDecodeSrc = 1000024001,
    VideoDecodeDpb = 1000024002,
    SharedPresent = 1000111000,
    FragmentDensityMapOptimal = 1000218000,
    FragmentShadingRateAttachmentOptimal = 1000164003,
    RenderingLocalRead = 1000232000,
    VideoEncodeDst = 1000299000,
    VideoEncodeSrc = 1000299001,
    VideoEncodeDpb = 1000299002,
    AttachmentFeedbackLoopOptimal = 1000339000,
    DepthReadOnlyStencilAttachmentOptimal = 1000117000,
    DepthAttachmentStencilReadOnlyOptimal = 1000117001,
    DepthAttachmentOptimal = 1000241000,
    DepthReadOnlyOptimal = 1000241001,
    StencilAttachmentOptimal = 1000241002,
    StencilReadOnlyOptimal = 1000241003,
    ReadOnlyOptimal = 1000314000,
    AttachmentOptimal = 1000314001,
}

public enum SampleCount : uint
{
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
    Sixteen = 16,
    ThirtyTwo = 32,
    SixtyFour = 64,
}

public enum MemoryLocation : uint
{
    Unknown = 0,
    GpuOnly = 1,
    CpuToGpu = 2,
    GpuToCpu = 3,
}

[Flags]
public enum BufferUsage : uint
{
    TransferSrc = 0x1,
    TransferDst = 0x2,
    UniformTexel = 0x4,
    StorageTexel = 0x8,
    Uniform = 0x10,
    Storage = 0x20,
    Index = 0x40,
    Vertex = 0x80,
    Indirect = 0x100,
    VideoDecodeSrc = 0x2000,
    VideoDecodeDst = 0x4000,
    TransformFeedback = 0x800,
    TransformFeedbackCounter = 0x1000,
    ConditionalRendering = 0x200,
    AccelerationStructureStorage = 0x100000,
    ShaderBindingTable = 0x400,
    VideoEncodeDst = 0x8000,
    VideoEncodeSrc = 0x10000,
    SamplerDescriptor = 0x200000,
    ResourceDescriptor = 0x400000,
    MicromapBuildInputReadOnly = 0x800000,
    MicromapStorage = 0x1000000,
    ShaderDeviceAddress = 0x20000,
}

[Flags]
public enum ImageUsage : uint
{
    TransferSrc = 0x1,
    TransferDst = 0x2,
    Sampled = 0x4,
    Storage = 0x8,
    ColorAttachment = 0x10,
    DepthStencilAttachment = 0x20,
    TransientAttachment = 0x40,
    InputAttachment = 0x80,
    VideoDecodeDst = 0x400,
    VideoDecodeSrc = 0x800,
    VideoDecodeDpb = 0x1000,
    FragmentDensityMap = 0x200,
    FragmentShadingRateAttachment = 0x100,
    HostTransfer = 0x400000,
    VideoEncodeDst = 0x2000,
    VideoEncodeSrc = 0x4000,
    VideoEncodeDpb = 0x8000,
    AttachmentFeedbackLoop = 0x80000,
    InvocationMask = 0x40000,
    SampleWeight = 0x100000,
    SampleBlockMatch = 0x200000,
}

[Flags]
public enum DomainFlag : uint
{
    None = 0,
    Host = 1 << 0,
    Present = 1 << 1,
    Graphics = 1 << 2,
    Compute = 1 << 3,
    Transfer = 1 << 4,
}

public static class AccessExtensions
{
    public static ImageLayout ToImageLayout(this Access access) => (ImageLayout)PInvoke.AccessImageLayout((ulong)access);
}

public readonly record struct DepthState(bool TestEnable, bool WriteEnable, CompareOp CompareOp)
{
    public static DepthState Default => new(false, false, CompareOp.Always);

    public static DepthState Testing(CompareOp compareOp) => new(true, true, compareOp);

    public static DepthState Less() => Testing(CompareOp.Less);

    public static DepthState Greater() => Testing(CompareOp.Greater);

    public static DepthState ReadOnly(CompareOp compareOp) => new(true, false, compareOp);

    internal VirDepthState ToNative() =>
        new()
        {
            TestEnable = this.TestEnable ? 1u : 0u,
            WriteEnable = this.WriteEnable ? 1u : 0u,
            CompareOp = (int)this.CompareOp,
        };
}

public readonly record struct Sampler(ulong Raw);

public readonly record struct SamplerInfo(
    Filter MagFilter,
    Filter MinFilter,
    MipmapMode MipmapMode,
    SamplerAddressMode AddressModeU,
    SamplerAddressMode AddressModeV,
    SamplerAddressMode AddressModeW
)
{
    public static SamplerInfo Default => Linear;

    public static SamplerInfo Linear => Filtered(Filter.Linear);

    public static SamplerInfo Nearest => Filtered(Filter.Nearest);

    private static SamplerInfo Filtered(Filter filter) =>
        new(
            filter,
            filter,
            MipmapMode.Linear,
            SamplerAddressMode.ClampToEdge,
            SamplerAddressMode.ClampToEdge,
            SamplerAddressMode.ClampToEdge
        );

    public SamplerInfo WithMagFilter(Filter filter) => this with { MagFilter = filter };

    public SamplerInfo WithMinFilter(Filter filter) => this with { MinFilter = filter };

    public SamplerInfo WithAddressMode(SamplerAddressMode mode) =>
        this with { AddressModeU = mode, AddressModeV = mode, AddressModeW = mode };

    internal VirSamplerInfo ToNative() =>
        new()
        {
            MagFilter = (int)this.MagFilter,
            MinFilter = (int)this.MinFilter,
            MipmapMode = (int)this.MipmapMode,
            AddressModeU = (int)this.AddressModeU,
            AddressModeV = (int)this.AddressModeV,
            AddressModeW = (int)this.AddressModeW,
        };
}

public readonly record struct BufferInfo(ulong Size, BufferUsage Usage, MemoryLocation Location, string Name = "", ulong Alignment = 1)
{
    public static BufferInfo Vertex(ulong size) => new(size, BufferUsage.Vertex, MemoryLocation.CpuToGpu);

    public static BufferInfo Staging(ulong size) => new(size, BufferUsage.TransferSrc, MemoryLocation.CpuToGpu);

    public BufferInfo WithUsage(BufferUsage usage) => this with { Usage = this.Usage | usage };

    public BufferInfo WithName(string name) => this with { Name = name };

    public BufferInfo WithAlignment(ulong alignment) => this with { Alignment = alignment };
}

public readonly record struct ImageInfo
{
    private enum ImageKind : uint
    {
        ColorTarget = 0,
        DepthTarget = 1,
        Texture = 2,
    }

    private readonly ImageKind kind;

    private ImageInfo(ImageKind kind, uint width, uint height, Format format)
    {
        this.kind = kind;
        this.Width = width;
        this.Height = height;
        this.Format = format;
        this.Name = "";
        this.Usage = 0;
        this.MipLevels = 1;
        this.ArrayLayers = 1;
        this.Samples = SampleCount.One;
    }

    public uint Width { get; init; }

    public uint Height { get; init; }

    public Format Format { get; init; }

    public ImageUsage Usage { get; init; }

    public uint MipLevels { get; init; }

    public uint ArrayLayers { get; init; }

    public SampleCount Samples { get; init; }

    public string Name { get; init; }

    public static ImageInfo ColorTarget(uint width, uint height, Format format) =>
        new(ImageKind.ColorTarget, width, height, format);

    public static ImageInfo DepthTarget(uint width, uint height, Format format) =>
        new(ImageKind.DepthTarget, width, height, format);

    public static ImageInfo Texture(uint width, uint height, Format format) =>
        new(ImageKind.Texture, width, height, format);

    public ImageInfo WithUsage(ImageUsage usage) => this with { Usage = this.Usage | usage };

    public ImageInfo WithName(string name) => this with { Name = name };

    public ImageInfo WithMips(uint mipLevels) => this with { MipLevels = mipLevels };

    public ImageInfo WithLayers(uint arrayLayers) => this with { ArrayLayers = arrayLayers };

    public ImageInfo WithSamples(SampleCount samples) => this with { Samples = samples };

    internal uint Kind => (uint)this.kind;
}

public readonly record struct TransientImageInfo(
    uint Width,
    uint Height,
    Format Format,
    ImageUsage Usage = 0,
    string Name = ""
)
{
    public uint Depth { get; init; } = 1;

    public uint MipLevels { get; init; } = 1;

    public uint ArrayLayers { get; init; } = 1;

    public SampleCount Samples { get; init; } = SampleCount.One;

    public TransientImageInfo WithUsage(ImageUsage usage) => this with { Usage = this.Usage | usage };

    public TransientImageInfo WithName(string name) => this with { Name = name };

    public TransientImageInfo WithDepth(uint depth) => this with { Depth = depth };

    public TransientImageInfo WithMips(uint mipLevels) => this with { MipLevels = mipLevels };

    public TransientImageInfo WithLayers(uint arrayLayers) => this with { ArrayLayers = arrayLayers };

    public TransientImageInfo WithSamples(SampleCount samples) => this with { Samples = samples };
}

public readonly record struct BindlessSet(uint Index, ulong Layout, ulong Set)
{
    internal VirBindlessSet ToNative() =>
        new()
        {
            Index = this.Index,
            Layout = this.Layout,
            Set = this.Set,
        };
}

public readonly record struct BufferCopy(ulong SrcOffset, ulong DstOffset, ulong Size);

public readonly record struct BufferImageCopy(
    uint Width,
    uint Height,
    ulong BufferOffset = 0,
    uint MipLevel = 0
)
{
    public int OffsetX { get; init; }

    public int OffsetY { get; init; }

    public int OffsetZ { get; init; }

    public uint Depth { get; init; } = 1;
}

public sealed unsafe class GpuBuffer
{
    private nint handle;

    internal GpuBuffer(nint handle) => this.handle = handle;

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(GpuBuffer));

    public ulong DeviceAddress => PInvoke.BufferDeviceAddress(this.Handle);

    public ulong Size => PInvoke.BufferSize(this.Handle);

    internal nint Release()
    {
        var handle = this.Handle;
        this.handle = 0;
        return handle;
    }

    public void Write<T>(ulong offset, ReadOnlySpan<T> data)
        where T : unmanaged
    {
        var bytes = MemoryMarshal.AsBytes(data);
        fixed (byte* ptr = bytes)
            VirException.ThrowIfFailed(PInvoke.BufferWrite(this.Handle, offset, ptr, (nuint)bytes.Length));
    }
}

public sealed class GpuImage
{
    private nint handle;

    internal GpuImage(nint handle) => this.handle = handle;

    internal nint Handle => this.handle != 0 ? this.handle : throw new ObjectDisposedException(nameof(GpuImage));

    public ulong ImageHandle => PInvoke.ImageHandle(this.Handle);

    public ulong ViewHandle => PInvoke.ImageViewHandle(this.Handle);

    internal nint Release()
    {
        var handle = this.Handle;
        this.handle = 0;
        return handle;
    }
}
