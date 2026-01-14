using System;
using System.Runtime.InteropServices;
using System.Windows.Media.Media3D;

namespace RawDxPlayerWpf.Processing
{
    public sealed class NativeImageProcessor : IImageProcessor
    {
        private bool _initialized;

        public void Initialize(int gpuId, IntPtr inoutDxSharedHandle)
        {
            int r = NativeImageProc.IPC_Init(gpuId, inoutDxSharedHandle);
            if (r != 0) throw new InvalidOperationException($"IPC_Init failed: {r}");
            _initialized = true;
        }

        public void SetParameters(object paramStruct)
        {
            if (!_initialized) return;

            if (paramStruct is NativeImageProc.IPC_Params p)
            {
                int r = NativeImageProc.IPC_SetParams(ref p);
                if (r != 0) throw new InvalidOperationException($"IPC_SetParams failed: {r}");
                return;
            }

            throw new ArgumentException("paramStruct must be NativeImageProc.IPC_Params");
        }

        public void Execute()
        {
            if (!_initialized) return;
            int r = NativeImageProc.IPC_Execute();
            if (r != 0) throw new InvalidOperationException($"IPC_Execute failed: {r}");
        }

        public void Shutdown()
        {
            if (!_initialized) return;
            NativeImageProc.IPC_Shutdown();
            _initialized = false;
        }
        
        // NEW: Readback RAW16 in native side (GPU->CPU copy in C Main DLL)
        public byte[] ReadbackRaw16(int width, int height)
        {
            if (!_initialized) return null;
            int bytes = checked(width * height * 2);
            var managed = new byte[bytes];

            GCHandle h = default;
            try
            {
                h = GCHandle.Alloc(managed, GCHandleType.Pinned);
                IntPtr p = h.AddrOfPinnedObject();
                int r = NativeImageProc.IPC_ReadbackRaw16(p, bytes);
                if (r != 0) throw new InvalidOperationException($"IPC_ReadbackRaw16 failed: {r}");
                return managed;
            }
            finally
            {
                if (h.IsAllocated) h.Free();
            }
        }
    }
}
