using System;
using System.Runtime.InteropServices;

namespace RawDxPlayerWpf.Processing
{
    internal static class NativeImageProc
    {
        // DLL名：出力された dll ファイル名に合わせる
        private const string DllName = "ImageProcCudaDll.dll";

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        internal struct IPC_Params
        {
            public uint sizeBytes;
            public uint version;

            public int window;
            public int level;

            public int enableEdge;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
            public int[] reserved;
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_Init(int gpuId, IntPtr inSharedHandle, IntPtr outSharedHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_SetParams(ref IPC_Params p);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_Execute();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_Shutdown();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_ReadbackRaw16(IntPtr dst, int dstBytes);


        internal static IPC_Params MakeDefaultParams(int window, int level, int enableEdge)
        {
            return new IPC_Params
            {
                sizeBytes = (uint)Marshal.SizeOf(typeof(IPC_Params)),
                version = 1,
                window = window,
                level = level,
                enableEdge = enableEdge,
                reserved = new int[8]
            };
        }
    }
}
