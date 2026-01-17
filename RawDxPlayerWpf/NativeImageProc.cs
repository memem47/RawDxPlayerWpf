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
            public int width;
            public int height;
            public uint sizeBytes;
            public uint version;

            public int window;
            public int level;

            public int enableEdge;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
            public int[] reserved;
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_Init(int gpuId, IntPtr ioSharedHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_InitWithIoBuffer(int gpuId, IntPtr ioBufferPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_SetParams(ref IPC_Params p);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_Execute();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_Shutdown();

        public static IPC_Params MakeDefaultParams(
            int width, int height,
            int window, int level,
            int enableEdge,
            int enableBlur,
            int enableInvert,
            int enableThreshold,
            int thresholdValue)
        {
            return new IPC_Params
            {
                sizeBytes = (uint)Marshal.SizeOf(typeof(IPC_Params)),
                version = 1,
                width = width,
                height = height,
                window = window,
                level = level,
                enableEdge = enableEdge,
                reserved = new int[8]
                {
                    enableBlur,      // reserved[0]
                    enableInvert,    // reserved[1]
                    enableThreshold, // reserved[2]
                    thresholdValue,  // reserved[3]
                    0,0,0,0
                }
            };
        }
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_UploadRaw16(IntPtr src, int srcBytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_UploadRaw16ToBuffer(IntPtr src, int srcBytes, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_ReadbackRaw16(IntPtr dst, int dstBytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_ReadbackRaw16FromBuffer(IntPtr dst, int dstBytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr IPC_CreateIoSharedHandle(int gpuId, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void IPC_DestroyIoSharedHandle(IntPtr sharedHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr IPC_CreateIoBuffer(int gpuId, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void IPC_ReleaseD3D11Resource(IntPtr d3d11Resource);

        // 失敗原因取得
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IPC_GetLastHr();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr IPC_GetLastErr();
    }
}
