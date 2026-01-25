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

        public void InitializeWithIoBuffer(int gpuId, IntPtr inoutDxSharedBuffer)
        {
            int r = NativeImageProc.IPC_InitWithIoBuffer(gpuId, inoutDxSharedBuffer);
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

        public void UploadRaw16(byte[] raw16, int w, int h)
        {
            if (!_initialized) throw new InvalidOperationException("Not initialized.");
            if (raw16 == null) throw new ArgumentNullException(nameof(raw16));
            int bytes = w * h * 2;
            if (raw16.Length < bytes) throw new ArgumentException("raw16 buffer too small.");

            GCHandle hnd = default;
            try
            {
                hnd = GCHandle.Alloc(raw16, GCHandleType.Pinned);
                IntPtr p = hnd.AddrOfPinnedObject();
                //int r = NativeImageProc.IPC_UploadRaw16(p, bytes);
                int r = NativeImageProc.IPC_UploadRaw16ToBuffer(p, bytes, w, h);
                if (r != 0) throw new InvalidOperationException($"IPC_UploadRaw16 failed: {r}");
            }
            finally
            {
                if (hnd.IsAllocated) hnd.Free();
            }
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
                //int r = NativeImageProc.IPC_ReadbackRaw16(p, bytes);
                int r = NativeImageProc.IPC_ReadbackRaw16FromBuffer(p, bytes);
                if (r != 0) throw new InvalidOperationException($"IPC_ReadbackRaw16 failed: {r}");
                return managed;
            }
            finally
            {
                if (h.IsAllocated) h.Free();
            }
        }

        public void UploadRaw16(int gpuId, IntPtr inoutDxSharedBuffer, byte[] raw16, int w, int h)
        {
            if (!_initialized) throw new InvalidOperationException("Not initialized.");
            if (raw16 == null) throw new ArgumentNullException(nameof(raw16));
            int bytes = w * h * 2;
            if (raw16.Length < bytes) throw new ArgumentException("raw16 buffer too small.");

            GCHandle hnd = default;
            try
            {
                hnd = GCHandle.Alloc(raw16, GCHandleType.Pinned);
                IntPtr p = hnd.AddrOfPinnedObject();
                int r = NativeImageProc.IPC_UploadRaw16ToBufferEx(gpuId, inoutDxSharedBuffer, p, bytes, w, h);
                if (r != 0) throw new InvalidOperationException($"IPC_UploadRaw16 failed: {r}");
            }
            finally
            {
                if (hnd.IsAllocated) hnd.Free();
            }
        }

        // NEW: Readback RAW16 in native side (GPU->CPU copy in C Main DLL)
        public byte[] ReadbackRaw16(int gpuId, IntPtr inoutDxSharedBuffer, int width, int height)
        {
            if (!_initialized) return null;
            int bytes = checked(width * height * 2);
            var managed = new byte[bytes];

            GCHandle h = default;
            try
            {
                h = GCHandle.Alloc(managed, GCHandleType.Pinned);
                IntPtr p = h.AddrOfPinnedObject();
                //int r = NativeImageProc.IPC_ReadbackRaw16(p, bytes);
                int r = NativeImageProc.IPC_ReadbackRaw16FromBufferEx(gpuId, inoutDxSharedBuffer, p, bytes, width, height);
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
