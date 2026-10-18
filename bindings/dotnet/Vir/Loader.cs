using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Vir;

internal static class Loader
{
    internal const string LogicalName = "vir_ffi";

    [SuppressMessage(
        "Usage",
        "CA2255:The 'ModuleInitializer' attribute should not be used in libraries",
        Justification = "The resolver must be registered before the first P/Invoke."
    )]
    [ModuleInitializer]
    internal static void Register()
    {
        NativeLibrary.SetDllImportResolver(typeof(Loader).Assembly, Resolve);
        Debug.Assert(NativeLayout.IsValid(), "vir-ffi structs do not match their native layout");
    }

    private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != LogicalName)
            return nint.Zero;

        var fileName = OperatingSystem.IsWindows() ? "vir_ffi.dll"
            : OperatingSystem.IsMacOS() ? "libvir_ffi.dylib"
            : "libvir_ffi.so";

        return NativeLibrary.Load(fileName, assembly, DllImportSearchPath.AssemblyDirectory);
    }
}
