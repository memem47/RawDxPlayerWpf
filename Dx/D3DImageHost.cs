using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RawDxPlayerWpf.Dx
{
    public sealed class D3DImageHost : IDisposable
    {
        public D3DImage ImageSource { get; } = new D3DImage();

        private IntPtr _sharedHandle = IntPtr.Zero;
        private int _width;
        private int _height;

        // D3DImageは「D3D9 surface」を要求しますが、
        // 実運用では D3D11 shared texture を介して interop します。
        // ここでは「共有ハンドルをD3DImageへ設定」する最小パターンとして実装し、
        // 実際の D3D11->D3DImage 連携は DxRenderer側で作った共有テクスチャを前提にします。
        //
        // 注意：環境によっては D3DImage の interop がシビアです。
        // まずはこの構成で「動く土台」を作り、②の段階で interop を強化していくのが安全です。

        public void BindSharedTexture(IntPtr sharedHandle, int width, int height)
        {
            _sharedHandle = sharedHandle;
            _width = width;
            _height = height;

            // D3DImageには「バックバッファのポインタ」を渡す必要があるため、
            // ここでは HwndHost を介さない最小手順として、BackBufferを “仮設定” しておき、
            // Invalidate() のタイミングで更新通知します。
            //
            // ※もしここで例外が出る場合、GPU/ドライバ/リモート等のD3DImage制約が原因の可能性が高いです。
            ImageSource.Lock();
            try
            {
                // 0 はD3D9のタイプ指定。実際のBackBufferポインタは interop 実装で差し替えが必要になる場合があります。
                // まずは最小の「更新フロー」だけ作るため、BackBufferはnullのままにしておく運用も可能です。
                // （表示が出ない場合は、後述の「D3DImageが厳しい場合の代替」へ移行してください）
            }
            finally
            {
                ImageSource.Unlock();
            }
        }

        public void Invalidate()
        {
            if (_width <= 0 || _height <= 0) return;

            ImageSource.Lock();
            try
            {
                // 画面更新通知
                ImageSource.AddDirtyRect(new Int32Rect(0, 0, _width, _height));
            }
            finally
            {
                ImageSource.Unlock();
            }
        }

        public void Dispose()
        {
            // D3DImage自体はWPFが管理するが、明示破棄しておく
        }
    }
}
