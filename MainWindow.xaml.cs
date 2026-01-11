using Microsoft.Win32;
using RawDxPlayerWpf.Dx;
using RawDxPlayerWpf.Processing;
using RawDxPlayerWpf.Raw;
using System;
using System.IO;
using System.Diagnostics;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.Windows.Media;

namespace RawDxPlayerWpf
{
    public partial class MainWindow : Window
    {
        private readonly DispatcherTimer _timer = new DispatcherTimer();

        // Perf (wall-clock time around _processor.Execute) : last 10 frames
        private readonly double[] _execMs = new double[10];
        private int _execMsCount = 0;
        private int _execMsIndex = 0;

        private RawSequence _seq;
        private int _index;

        private DxRenderer _renderer;
        private WriteableBitmap _wb;

        private readonly IImageProcessor _processor = new NativeImageProcessor();
        private bool _dllInitialized;

        // Save Result (step1)
        // できれば間引き、保存上限、非同期化を追加
        private string _outputDir;

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

            // とりえあず出力先フォルダ名を固定
            _outputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
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

            // ① RAW16をそのまま読み込んで input(R16) に upload
            byte[] raw16 = RawFrameReader.Load16RawBytes(path, _seq.Width, _seq.Height);
            _renderer.UploadInputRaw16(raw16);

            bool dllOn = (CbDllOn.IsChecked == true);

            if (dllOn && _dllInitialized)
            {
                // ★注意：この段階では DLL 側がまだ BGRA input 前提の可能性が高いです。
                // B(=DLL側R16対応)が終わるまで DLL ON は使わないのが安全です。
                ApplyParamsToDll();
                var sw = Stopwatch.StartNew();
                _processor.Execute();
                sw.Stop();

                PushExecMs(sw.Elapsed.TotalMilliseconds);
                UpdatePerfText();
            }
            else
            {
                // ② DLL OFF の場合はCPUでWL/WWして output(BGRA) に書く
                byte[] bgra = RawFrameReader.Convert16ToBgra8(raw16, _seq.Width, _seq.Height, _window, _level);
                _renderer.UploadOutputBgra(bgra);

                // DLL OFF のときは perf を更新しない（必要ならここも計測可）
            }

            // ③ 出力を readback して表示
            var bgraOut = _renderer.ReadbackOutputBgra();
            _wb.WritePixels(new Int32Rect(0, 0, _seq.Width, _seq.Height), bgraOut, _seq.Width * 4, 0);

            TbStatus.Text = $"Frame {_index + 1}/{_seq.Files.Count} : {System.IO.Path.GetFileName(path)}";

            // 保存
            SaveFrameIfEnabled(bgraOut, _seq.Width, _seq.Height, _index);
        }
        
        private void SaveFrameIfEnabled(byte[] bgraOut, int width, int height, int frameIndex)
        {
            if (CbSaveOn?.IsChecked != true) return;
            if (bgraOut == null || bgraOut.Length < width * height * 4) return;

            try
            {
                Directory.CreateDirectory(_outputDir);

                // 連番（わかりやすく固定桁）
                string stem = $"frame_{frameIndex:D6}_proc";
                string pngPath = Path.Combine(_outputDir, stem + ".png");
                string rawPath = Path.Combine(_outputDir, stem + ".raw");

                SaveBgraToPng(bgraOut, width, height, pngPath);
                SaveBgraAsRaw16LittleEndian(bgraOut, width, height, rawPath);
            }
            catch (Exception ex)
            {
                // 毎フレームMessageBoxは地獄なのでステータスだけ更新
                UpdateStatus($"Save failed: {ex.Message}");
            }
        }

        private static void SaveBgraToPng(byte[] bgra, int width, int height, string outPath)
        {
            var bs = BitmapSource.Create(
                width, height,
                96, 96,
                PixelFormats.Bgra32,
                null,
                bgra,
                width * 4);

            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bs));
            using (var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                enc.Save(fs);
            }
        }

        private static void SaveBgraAsRaw16LittleEndian(byte[] bgra, int width, int height, string outPath)
        {
            int pixels = checked(width * height);
            byte[] raw16 = new byte[pixels * 2];

            for (int i = 0; i < pixels; i++)
            {
                // BGRAのB成分を灰度として使用（R=G=BなのでどれでもOK）
                byte g8 = bgra[i * 4 + 0];
                ushort v16 = (ushort)(g8 * 257); // 0..255 -> 0..65535
                raw16[i * 2 + 0] = (byte)(v16 & 0xFF);
                raw16[i * 2 + 1] = (byte)(v16 >> 8);
            }

            File.WriteAllBytes(outPath, raw16);
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

        private void PushExecMs(double ms)
        {
            _execMs[_execMsIndex] = ms;
            _execMsIndex = (_execMsIndex + 1) % _execMs.Length;
            _execMsCount = Math.Min(_execMsCount + 1, _execMs.Length);
        }

        private double GetAvgExecMs()
        {
            if (_execMsCount == 0) return 0;
            double sum = 0;
            for (int i = 0; i<_execMsCount; i++) sum += _execMs[i];
            return sum / _execMsCount;
        }

        private void UpdatePerfText()
        {
            double avg = GetAvgExecMs();
            if (avg <= 0)
            {
                TbCudaMs.Text = "CUDA avg(10): -";
                return;
            }
            double fps = 1000.0 / avg;
            TbCudaMs.Text = $"CUDA avg(10): {avg:F2} ms  ({fps:F1} fps)";
        }
    }
}
