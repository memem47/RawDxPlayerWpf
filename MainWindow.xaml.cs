using Microsoft.Win32;
using RawDxPlayerWpf.Dx;
using RawDxPlayerWpf.Processing;
using RawDxPlayerWpf.Raw;
using System;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace RawDxPlayerWpf
{
    public partial class MainWindow : Window
    {
        private readonly DispatcherTimer _timer = new DispatcherTimer();

        private RawSequence _seq;
        private int _index;

        private DxRenderer _renderer;
        private WriteableBitmap _wb;

        private readonly IImageProcessor _processor = new NativeImageProcessor();
        private bool _dllInitialized;

        // params
        private int _window = 4000;
        private int _level = 2000;
        private int _enableEdge = 0;

        private bool _suppressParamEvents;

        private bool _uiReady = false;

        public MainWindow()
        {
            InitializeComponent();
            _timer.Tick += (_, __) => RenderNextFrame();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            _suppressParamEvents = true;
            SyncGuiFromParams();
            _suppressParamEvents = false;

            _uiReady = true; // ★ここで初期化完了
            UpdateStatus("Ready");
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            StopPlayback();

            try
            {
                if (_dllInitialized)
                {
                    _processor.Shutdown();
                    _dllInitialized = false;
                }
            }
            catch { /* ignore */ }

            _renderer?.Dispose();
        }

        private void BtnOpen_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Title = "RAWフォルダ内の任意のRAW/BINファイルを選択してください",
                Filter = "RAW/BIN files (*.raw;*.bin)|*.raw;*.bin",
                CheckFileExists = true
            };

            if (dlg.ShowDialog() != true) return;

            LoadSequenceFromFile(dlg.FileName);
        }

        private void LoadSequenceFromFile(string filePath)
        {
            StopPlayback();

            _seq = RawSequence.FromAnyFileInFolder(filePath);
            _index = 0;

            // renderer / writeablebitmap
            _renderer?.Dispose();
            _renderer = new DxRenderer(_seq.Width, _seq.Height, gpuId: 0);

            _wb = new WriteableBitmap(_seq.Width, _seq.Height, 96, 96,
                System.Windows.Media.PixelFormats.Bgra32, null);
            ImgView.Source = _wb;

            // init dll
            try
            {
                int gpuId = 0;
                _processor.Initialize(gpuId, _renderer.InputSharedHandle, _renderer.OutputSharedHandle);
                _dllInitialized = true;
                ApplyParamsToDll();
            }
            catch (Exception ex)
            {
                _dllInitialized = false;
                UpdateStatus($"DLL init failed: {ex.Message}");
            }

            UpdateStatus($"Loaded: {_seq.Files.Count} frames, {_seq.Width}x{_seq.Height}");
            RenderNextFrame(forceSameIndex: true);
        }

        private void BtnPlay_Click(object sender, RoutedEventArgs e)
        {
            if (_seq == null)
            {
                MessageBox.Show("先にOpenしてください。", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            if (!int.TryParse(TbFps.Text, out int fps) || fps <= 0) fps = 30;
            _timer.Interval = TimeSpan.FromMilliseconds(1000.0 / fps);
            _timer.Start();

            UpdateStatus("Playing...");
        }

        private void BtnStop_Click(object sender, RoutedEventArgs e) => StopPlayback();

        private void StopPlayback()
        {
            _timer.Stop();
            if (_seq != null) UpdateStatus("Stopped.");
        }

        private void RenderNextFrame(bool forceSameIndex = false)
        {
            if (_seq == null || _renderer == null) return;

            if (!forceSameIndex)
            {
                _index++;
                if (_index >= _seq.Files.Count) _index = 0;
            }

            string path = _seq.Files[_index];

            // 現状は CPU で WL/WW して BGRA を作って input に upload
            byte[] bgra = RawFrameReader.Load16ToBgra8(path, _seq.Width, _seq.Height, _window, _level);
            _renderer.UploadInputBgra(bgra);

            bool dllOn = (CbDllOn.IsChecked == true);

            if (dllOn && _dllInitialized)
            {
                // GUI変更を DLL へ反映（軽ければ毎フレームでもOK）
                ApplyParamsToDll();
                _processor.Execute();
            }
            else
            {
                // DLL OFF: input を output にコピーして表示
                _renderer.PassthroughCopyInputToOutput();
            }

            // output readback -> WriteableBitmap
            byte[] bgraOut = _renderer.ReadbackOutputBgra();
            _wb.WritePixels(new Int32Rect(0, 0, _seq.Width, _seq.Height), bgraOut, _seq.Width * 4, 0);

            TbStatus.Text = $"Frame {_index + 1}/{_seq.Files.Count} : {System.IO.Path.GetFileName(path)}";
        }

        // ---- GUI events: Params ----

        private void BtnResetParams_Click(object sender, RoutedEventArgs e)
        {
            _window = 4000;
            _level = 2000;
            _enableEdge = 0;
            SyncGuiFromParams();
            ApplyParamsToDll();

            if (_seq != null) RenderNextFrame(forceSameIndex: true);
        }

        private void CbDllOn_Checked(object sender, RoutedEventArgs e)
        {
            ApplyParamsToDll();
            if (_seq != null) RenderNextFrame(forceSameIndex: true);
        }

        private void CbDllOn_Unchecked(object sender, RoutedEventArgs e)
        {
            if (_seq != null) RenderNextFrame(forceSameIndex: true);
        }

        private void CbEdge_Checked(object sender, RoutedEventArgs e)
        {
            _enableEdge = 1;
            ApplyParamsToDll();
        }

        private void CbEdge_Unchecked(object sender, RoutedEventArgs e)
        {
            _enableEdge = 0;
            ApplyParamsToDll();
        }

        // Slider -> TextBox
        private void SlWindow_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!_uiReady) return;            // ★追加
            if (_suppressParamEvents) return;

            _window = (int)Math.Round(SlWindow.Value);
            if (TbWindow != null) TbWindow.Text = _window.ToString();
            ApplyParamsToDll();
        }

        private void SlLevel_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!_uiReady) return;            // ★追加
            if (_suppressParamEvents) return;
            _level = (int)Math.Round(SlLevel.Value);
            if (TbLevel != null) TbLevel.Text = _level.ToString();
            ApplyParamsToDll();
        }

        // TextBox -> Slider（入力確定）
        private void TbWindow_LostFocus(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(TbWindow.Text, out int v)) v = _window;
            v = Math.Max(1, Math.Min(65535, v));
            _window = v;

            _suppressParamEvents = true;
            SlWindow.Value = _window;
            _suppressParamEvents = false;

            ApplyParamsToDll();
        }

        private void TbLevel_LostFocus(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(TbLevel.Text, out int v)) v = _level;
            v = Math.Max(0, Math.Min(65535, v));
            _level = v;

            _suppressParamEvents = true;
            SlLevel.Value = _level;
            _suppressParamEvents = false;

            ApplyParamsToDll();
        }

        private void SyncGuiFromParams()
        {
            _suppressParamEvents = true;

            SlWindow.Value = _window;
            SlLevel.Value = _level;
            TbWindow.Text = _window.ToString();
            TbLevel.Text = _level.ToString();
            CbEdge.IsChecked = (_enableEdge != 0);

            _suppressParamEvents = false;
        }

        private void ApplyParamsToDll()
        {
            if (!_dllInitialized) return;

            try
            {
                var p = NativeImageProc.MakeDefaultParams(window: _window, level: _level, enableEdge: _enableEdge);
                _processor.SetParameters(p);
            }
            catch (Exception ex)
            {
                UpdateStatus($"IPC_SetParams failed: {ex.Message}");
            }
        }

        private void UpdateStatus(string msg) => TbStatus.Text = msg;
    }
}
