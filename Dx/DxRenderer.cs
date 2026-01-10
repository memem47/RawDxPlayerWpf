using System;
using SharpDX;
using SharpDX.Direct3D;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using Device = SharpDX.Direct3D11.Device;

namespace RawDxPlayerWpf.Dx
{
    public sealed class DxRenderer : IDisposable
    {
        private readonly Device _device;
        private readonly DeviceContext _ctx;

        // 共有テクスチャ（WPF表示・将来CUDA処理の入出力として使う）
        private readonly Texture2D _sharedTexture;
        private readonly IntPtr _sharedHandle;

        // CPU書き込み用（毎フレームMapするのはこっち）
        private readonly Texture2D _stagingTexture;

        private readonly int _width;
        private readonly int _height;

        public IntPtr SharedTextureHandle => _sharedHandle;

        public DxRenderer(int width, int height)
        {
            _width = width;
            _height = height;

            var flags = DeviceCreationFlags.BgraSupport;
            _device = new Device(DriverType.Hardware, flags);
            _ctx = _device.ImmediateContext;

            // 1) 共有テクスチャ（GPU側、CPUは直接Mapしない）
            var sharedDesc = new Texture2DDescription
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

                // 将来、CUDA/DX interop で共有するため
                OptionFlags = ResourceOptionFlags.Shared
            };

            _sharedTexture = new Texture2D(_device, sharedDesc);
            using (var resource = _sharedTexture.QueryInterface<SharpDX.DXGI.Resource>())
            {
                _sharedHandle = resource.SharedHandle;
            }

            // 2) CPU書き込み用ステージングテクスチャ
            var stagingDesc = new Texture2DDescription
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
            };

            _stagingTexture = new Texture2D(_device, stagingDesc);
        }

        // bgra: width*height*4 bytes
        public void UpdateFrame(byte[] bgra)
        {
            if (bgra == null) throw new ArgumentNullException(nameof(bgra));
            if (bgra.Length != _width * _height * 4)
                throw new ArgumentException("Invalid frame size", nameof(bgra));

            // staging を Map して CPU で書く
            var box = _ctx.MapSubresource(
                _stagingTexture, 0,
                MapMode.Write,
                SharpDX.Direct3D11.MapFlags.None);

            try
            {
                using (var stream = new DataStream(box.DataPointer, box.RowPitch * _height, true, true))
                {
                    int srcStride = _width * 4;
                    int dstStride = box.RowPitch;

                    int srcOffset = 0;
                    for (int y = 0; y < _height; y++)
                    {
                        stream.Position = y * dstStride;
                        stream.Write(bgra, srcOffset, srcStride);
                        srcOffset += srcStride;
                    }
                }
            }
            finally
            {
                _ctx.UnmapSubresource(_stagingTexture, 0);
            }

            // staging → shared にGPUコピー
            _ctx.CopyResource(_stagingTexture, _sharedTexture);
        }

        public void Dispose()
        {
            _stagingTexture?.Dispose();
            _sharedTexture?.Dispose();
            _ctx?.Dispose();
            _device?.Dispose();
        }
    }
}
