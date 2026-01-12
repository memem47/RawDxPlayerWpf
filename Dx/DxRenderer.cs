using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using System;
using System.Runtime.InteropServices;
using Device = SharpDX.Direct3D11.Device;

namespace RawDxPlayerWpf.Dx
{
    /// <summary>
    /// Shared texture (single IO):
    /// - IO: R16_UINT (upload RAW16, process in-place, readback RAW16)
    /// Display is done by CPU converting RAW16 -> BGRA8 (WL/WW).
    /// </summary>
    public sealed class DxRenderer : IDisposable
    {
        private readonly Device _device;
        private readonly DeviceContext _ctx;

        private readonly int _width;
        private readonly int _height;

        private readonly Texture2D _ioSharedTex;     // R16_UINT
        private readonly IntPtr _ioSharedHandle;

        // Staging upload/readback (CPU)
        private readonly Texture2D _stagingUploadRaw16;
        private readonly Texture2D _stagingReadbackRaw16;

        public IntPtr IoSharedHandle => _ioSharedHandle;

        public DxRenderer(int width, int height, int gpuId = 0)
        {
            _width = width;
            _height = height;

            var factory = new Factory1();
            var adapter = factory.GetAdapter1(gpuId);

            _device = new Device(adapter, DeviceCreationFlags.BgraSupport);
            _ctx = _device.ImmediateContext;

            adapter.Dispose();
            factory.Dispose();

            // ---- Shared IO (R16_UINT) ----
            var ioDesc = new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.R16_UInt,
                SampleDescription = new SampleDescription(1, 0),

                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.ShaderResource | BindFlags.UnorderedAccess,
                CpuAccessFlags = CpuAccessFlags.None,
                OptionFlags = ResourceOptionFlags.Shared
            };
            _ioSharedTex = new Texture2D(_device, ioDesc);
            using (var r = _ioSharedTex.QueryInterface<SharpDX.DXGI.Resource>())
                _ioSharedHandle = r.SharedHandle;

            // ---- Staging upload for RAW16 ----
            _stagingUploadRaw16 = new Texture2D(_device, new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.R16_UInt,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Staging,
                BindFlags = BindFlags.None,
                CpuAccessFlags = CpuAccessFlags.Write,
                OptionFlags = ResourceOptionFlags.None
            });

            // ---- Staging readback for RAW16 ----
            _stagingReadbackRaw16 = new Texture2D(_device, new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.R16_UInt,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Staging,
                BindFlags = BindFlags.None,
                CpuAccessFlags = CpuAccessFlags.Read,
                OptionFlags = ResourceOptionFlags.None
            });
        }

        /// <summary>
        /// Upload RAW16 (little endian) into shared IO texture.
        /// </summary>
        public void UploadIoRaw16(byte[] raw16LittleEndian)
        {
            if (raw16LittleEndian == null) throw new ArgumentNullException(nameof(raw16LittleEndian));
            int bytesNeeded = checked(_width * _height * 2);
            if (raw16LittleEndian.Length < bytesNeeded)
                throw new ArgumentException("raw16 buffer is too small", nameof(raw16LittleEndian));

            var box = _ctx.MapSubresource(_stagingUploadRaw16, 0, MapMode.Write, SharpDX.Direct3D11.MapFlags.None);
            try
            {
                int srcStride = _width * 2;
                int dstStride = box.RowPitch;

                for (int y = 0; y < _height; y++)
                {
                    int srcOffset = y * srcStride;
                    IntPtr dstPtr = box.DataPointer + y * dstStride;
                    Marshal.Copy(raw16LittleEndian, srcOffset, dstPtr, srcStride);
                }
            }
            finally
            {
                _ctx.UnmapSubresource(_stagingUploadRaw16, 0);
            }

            _ctx.CopyResource(_stagingUploadRaw16, _ioSharedTex);
        }

        /// <summary>
        /// Read back shared IO texture (R16_UINT) to CPU bytes.
        /// Returns exactly width*height*2 bytes.
        /// </summary>
        public byte[] ReadbackIoRaw16()
        {
            _ctx.Flush();
            _ctx.CopyResource(_ioSharedTex, _stagingReadbackRaw16);

            var box = _ctx.MapSubresource(_stagingReadbackRaw16, 0, MapMode.Read, SharpDX.Direct3D11.MapFlags.None);
            try
            {
                byte[] raw16 = new byte[_width * _height * 2];

                int dstStride = _width * 2;
                int srcStride = box.RowPitch;

                for (int y = 0; y < _height; y++)
                {
                    IntPtr srcPtr = box.DataPointer + y * srcStride;
                    int dstOffset = y * dstStride;
                    Marshal.Copy(srcPtr, raw16, dstOffset, dstStride);
                }

                return raw16;
            }
            finally
            {
                _ctx.UnmapSubresource(_stagingReadbackRaw16, 0);
            }
        }

        public void Dispose()
        {
            _stagingReadbackRaw16?.Dispose();
            _stagingUploadRaw16?.Dispose();
            _ioSharedTex?.Dispose();
            _ctx?.Dispose();
            _device?.Dispose();
        }
    }
}
