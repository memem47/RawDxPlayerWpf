using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using System;
using System.Runtime.InteropServices;
using Device = SharpDX.Direct3D11.Device;

namespace RawDxPlayerWpf.Dx
{
    /// <summary>
    /// Shared textures:
    /// - Input:  R16_UINT  (RAW16 upload)
    /// - Output: BGRA8_UNORM (display)
    /// </summary>
    public sealed class DxRenderer : IDisposable
    {
        private readonly Device _device;
        private readonly DeviceContext _ctx;

        private readonly int _width;
        private readonly int _height;

        // Shared textures
        private readonly Texture2D _inputSharedTex;   // R16
        private readonly Texture2D _outputSharedTex;  // BGRA
        private readonly IntPtr _inputSharedHandle;
        private readonly IntPtr _outputSharedHandle;

        // Staging for CPU upload/readback
        private readonly Texture2D _stagingUploadRaw16; // R16 staging (CPU->GPU)
        private readonly Texture2D _stagingUploadBgra;  // BGRA staging (CPU->GPU to output)
        private readonly Texture2D _stagingReadback;    // BGRA staging (GPU->CPU)

        public IntPtr InputSharedHandle => _inputSharedHandle;
        public IntPtr OutputSharedHandle => _outputSharedHandle;

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

            // ---- Shared Input (R16_UINT) ----
            var inDesc = new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.R16_UInt,
                SampleDescription = new SampleDescription(1, 0),

                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.ShaderResource,
                CpuAccessFlags = CpuAccessFlags.None,
                OptionFlags = ResourceOptionFlags.Shared
            };
            _inputSharedTex = new Texture2D(_device, inDesc);
            using (var r = _inputSharedTex.QueryInterface<SharpDX.DXGI.Resource>())
                _inputSharedHandle = r.SharedHandle;

            // ---- Shared Output (BGRA8) ----
            var outDesc = new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.B8G8R8A8_UNorm,
                SampleDescription = new SampleDescription(1, 0),

                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.ShaderResource | BindFlags.RenderTarget,
                CpuAccessFlags = CpuAccessFlags.None,
                OptionFlags = ResourceOptionFlags.Shared
            };
            _outputSharedTex = new Texture2D(_device, outDesc);
            using (var r = _outputSharedTex.QueryInterface<SharpDX.DXGI.Resource>())
                _outputSharedHandle = r.SharedHandle;

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

            // ---- Staging upload for BGRA (when DLL OFF) ----
            _stagingUploadBgra = new Texture2D(_device, new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.B8G8R8A8_UNorm,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Staging,
                BindFlags = BindFlags.None,
                CpuAccessFlags = CpuAccessFlags.Write,
                OptionFlags = ResourceOptionFlags.None
            });

            // ---- Staging readback for BGRA ----
            _stagingReadback = new Texture2D(_device, new Texture2DDescription
            {
                Width = width,
                Height = height,
                ArraySize = 1,
                MipLevels = 1,
                Format = Format.B8G8R8A8_UNorm,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Staging,
                BindFlags = BindFlags.None,
                CpuAccessFlags = CpuAccessFlags.Read,
                OptionFlags = ResourceOptionFlags.None
            });
        }

        /// <summary>
        /// Upload RAW16 (little endian) to shared input (R16_UINT).
        /// raw16 length must be >= width*height*2
        /// </summary>
        public void UploadInputRaw16(byte[] raw16LittleEndian)
        {
            if (raw16LittleEndian == null) throw new ArgumentNullException(nameof(raw16LittleEndian));
            int bytesNeeded = checked(_width * _height * 2);
            if (raw16LittleEndian.Length < bytesNeeded)
                throw new ArgumentException("raw16 buffer is too small", nameof(raw16LittleEndian));

            var box = _ctx.MapSubresource(_stagingUploadRaw16, 0, SharpDX.Direct3D11.MapMode.Write, SharpDX.Direct3D11.MapFlags.None);
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

            _ctx.CopyResource(_stagingUploadRaw16, _inputSharedTex);
        }

        /// <summary>
        /// Upload BGRA8 to shared output (BGRA8_UNORM). (Used when DLL OFF)
        /// </summary>
        public void UploadOutputBgra(byte[] bgra)
        {
            if (bgra == null) throw new ArgumentNullException(nameof(bgra));
            int bytesNeeded = checked(_width * _height * 4);
            if (bgra.Length < bytesNeeded)
                throw new ArgumentException("bgra buffer is too small", nameof(bgra));

            var box = _ctx.MapSubresource(_stagingUploadBgra, 0, MapMode.Write, SharpDX.Direct3D11.MapFlags.None);
            try
            {
                int srcStride = _width * 4;
                int dstStride = box.RowPitch;

                for (int y = 0; y < _height; y++)
                {
                    int srcOffset = y * srcStride;
                    IntPtr dstPtr = box.DataPointer + y * dstStride;

                    Marshal.Copy(bgra, srcOffset, dstPtr, srcStride);
                }
            }
            finally
            {
                _ctx.UnmapSubresource(_stagingUploadBgra, 0);
            }

            _ctx.CopyResource(_stagingUploadBgra, _outputSharedTex);
        }

        /// <summary>
        /// Readback shared output (BGRA8) to CPU.
        /// </summary>
        public byte[] ReadbackOutputBgra()
        {
            _ctx.Flush();
            _ctx.CopyResource(_outputSharedTex, _stagingReadback);

            var box = _ctx.MapSubresource(_stagingReadback, 0, MapMode.Read, SharpDX.Direct3D11.MapFlags.None);
            try
            {
                byte[] bgra = new byte[_width * _height * 4];

                int dstStride = _width * 4;
                int srcStride = box.RowPitch;

                for (int y = 0; y < _height; y++)
                {
                    IntPtr srcPtr = box.DataPointer + y * srcStride;
                    int dstOffset = y * dstStride;

                    Marshal.Copy(srcPtr, bgra, dstOffset, dstStride);
                }

                return bgra;
            }
            finally
            {
                _ctx.UnmapSubresource(_stagingReadback, 0);
            }
        }

        public void Dispose()
        {
            _stagingReadback?.Dispose();
            _stagingUploadBgra?.Dispose();
            _stagingUploadRaw16?.Dispose();

            _outputSharedTex?.Dispose();
            _inputSharedTex?.Dispose();

            _ctx?.Dispose();
            _device?.Dispose();
        }
    }
}
