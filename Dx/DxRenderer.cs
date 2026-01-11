using SharpDX;
using SharpDX.Direct3D;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using System;
using System.Windows.Controls;
using Device = SharpDX.Direct3D11.Device;

namespace RawDxPlayerWpf.Dx
{
    public sealed class DxRenderer : IDisposable
    {
        private readonly Device _device;
        private readonly DeviceContext _ctx;

        // 入力（CPU→GPU）
        private readonly Texture2D _inputSharedTex;
        private readonly IntPtr _inputSharedHandle;

        // 出力（GPU処理結果：②がここへ書く）
        private readonly Texture2D _outputSharedTex;
        private readonly IntPtr _outputSharedHandle;

        // CPU書き込み用（inputに転送する）
        private readonly Texture2D _stagingUpload;

        // CPU読み戻し用（outputから読む）
        private readonly Texture2D _stagingReadback;

        private readonly int _width;
        private readonly int _height;

        public IntPtr InputSharedHandle => _inputSharedHandle;
        public IntPtr OutputSharedHandle => _outputSharedHandle;

        public DxRenderer(int width, int height, int gpuId)
        {
            _width = width;
            _height = height;

            var factory = new Factory1();
            var adapter = factory.GetAdapter1(gpuId);   // gpuId を使う
            _device = new Device(adapter, DeviceCreationFlags.BgraSupport);
            _ctx = _device.ImmediateContext;

            // factory/adapter は Dispose してOK（Deviceが保持）
            adapter.Dispose();
            factory.Dispose();

            // 共有テクスチャ（Default + Shared）
            Texture2DDescription SharedDesc() => new Texture2DDescription
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

            _inputSharedTex = new Texture2D(_device, SharedDesc());
            using (var r = _inputSharedTex.QueryInterface<SharpDX.DXGI.Resource>())
                _inputSharedHandle = r.SharedHandle;

            _outputSharedTex = new Texture2D(_device, SharedDesc());
            using (var r = _outputSharedTex.QueryInterface<SharpDX.DXGI.Resource>())
                _outputSharedHandle = r.SharedHandle;

            // CPU→GPU upload（Mapするのはこっち）
            _stagingUpload = new Texture2D(_device, new Texture2DDescription
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

            // GPU→CPU readback（outputを読む）
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

        // inputSharedTex を更新
        public void UploadInputBgra(byte[] bgra)
        {
            if (bgra == null) throw new ArgumentNullException(nameof(bgra));
            if (bgra.Length != _width * _height * 4)
                throw new ArgumentException("Invalid frame size", nameof(bgra));

            var box = _ctx.MapSubresource(_stagingUpload, 0, MapMode.Write, SharpDX.Direct3D11.MapFlags.None);
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
                _ctx.UnmapSubresource(_stagingUpload, 0);
            }

            _ctx.CopyResource(_stagingUpload, _inputSharedTex);
        }

        // いまは「処理なし」なので input→output へコピー（②実装後は不要）
        public void PassthroughCopyInputToOutput()
        {
            _ctx.CopyResource(_inputSharedTex, _outputSharedTex);
        }

        // outputSharedTex をCPUへ読み戻し（表示用）
        public byte[] ReadbackOutputBgra()
        {
            _ctx.Flush();  // ★追加（コピー前でも後でもOK、まずは前）
            _ctx.CopyResource(_outputSharedTex, _stagingReadback);

            var box = _ctx.MapSubresource(_stagingReadback, 0, MapMode.Read, SharpDX.Direct3D11.MapFlags.None);
            try
            {
                byte[] bgra = new byte[_width * _height * 4];
                using (var stream = new DataStream(box.DataPointer, box.RowPitch * _height, true, true))
                {
                    int dstStride = _width * 4;
                    int srcStride = box.RowPitch;

                    int dstOffset = 0;
                    for (int y = 0; y < _height; y++)
                    {
                        stream.Position = y * srcStride;
                        stream.Read(bgra, dstOffset, dstStride);
                        dstOffset += dstStride;
                    }
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
            _stagingUpload?.Dispose();

            _outputSharedTex?.Dispose();
            _inputSharedTex?.Dispose();

            _ctx?.Dispose();
            _device?.Dispose();
        }
    }
}
