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
        private int _enableBlur = 0;
        private int _enableInvert = 0;
        private int _enableThreshold = 0;
        private int _thresholdValue = 20000;

        private bool _suppressParamEvents;
        private bool _uiReady = false;

        // export
        private string _outputDir;

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

            _uiReady = true;
            UpdateStatus("Ready");

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

            _renderer?.Dispose();
            _renderer = new DxRenderer(_seq.Width, _seq.Height, gpuId: 0);

            _wb = new WriteableBitmap(_seq.Width, _seq.Height, 96, 96,
                PixelFormats.Bgra32, null);
            ImgView.Source = _wb;

            // init dll (single IO handle)
            try
            {
                int gpuId = 0;
                _processor.Initialize(gpuId, _renderer.IoSharedHandle);
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

        /// <summary>
        /// Pipeline (single IO texture, true 16-bit):
        /// 1) Load RAW16 -> Upload to IO (R16)
        /// 2) If DLL ON: in-place processing IO->IO using intermediate arrays in CUDA
        /// 3) Readback IO RAW16 -> CPU WL/WW -> BGRA8 -> display
        /// 4) Optional export (RAW16 + PNG)
        /// </summary>
        private void RenderNextFrame(bool forceSameIndex = false)
        {
            if (_seq == null || _renderer == null) return;

            var swTotal = Stopwatch.StartNew();
            long tLoadUp = 0, tCuda = 0, tRead = 0, tDisp = 0;

            if (!forceSameIndex)
            {
                _index++;
                if (_index >= _seq.Files.Count) _index = 0;
            }

            string path = _seq.Files[_index];

            // ① RAW16 -> IO upload
            var sw = Stopwatch.StartNew();
            
            byte[] raw16In = RawFrameReader.Load16RawBytes(path, _seq.Width, _seq.Height);
            _renderer.UploadIoRaw16(raw16In);
            sw.Stop();
            tLoadUp = sw.ElapsedMilliseconds;

            // ② in-place processing (optional)
            bool dllOn = (CbDllOn.IsChecked == true);
            if (dllOn && _dllInitialized)
            {
                ApplyParamsToDll();
                sw.Restart();
                _processor.Execute();
                sw.Stop();
                tCuda = sw.ElapsedMilliseconds;
            }

            // ③ IO readback -> display conversion
            sw.Restart();
            byte[] raw16Out = _renderer.ReadbackIoRaw16();
            sw.Stop();
            tRead = sw.ElapsedMilliseconds;
    
            // ---- Display WL/WW ----
            // Edge出力は輝度分布が変わるので、表示用WL/WWは自動推定する
            int dispWindow = _window;
            int dispLevel  = _level;
            if (_enableEdge != 0)
            {
                GetAutoWindowLevelFromRaw16(raw16Out, out dispWindow, out dispLevel);
            }
    
            sw.Restart();
            byte[] bgra = RawFrameReader.Convert16ToBgra8(raw16Out, _seq.Width, _seq.Height, dispWindow, dispLevel);
            _wb.WritePixels(new Int32Rect(0, 0, _seq.Width, _seq.Height), bgra, _seq.Width * 4, 0);
            sw.Stop();
            tDisp = sw.ElapsedMilliseconds;

            swTotal.Stop();
            TbStatus.Text =
            $"Frame {_index + 1}/{_seq.Files.Count} : {System.IO.Path.GetFileName(path)}" +
            $" | Load+Up {tLoadUp}ms | CUDA {tCuda}ms | Read {tRead}ms | Disp {tDisp}ms | Total {swTotal.ElapsedMilliseconds}ms";

            // ④ export
            SaveFrameIfEnabled(raw16Out, bgra, _seq.Width, _seq.Height, _index);
        }

        // Edge表示用：RAW16のmin/maxからwindow/levelを推定
        private static void GetAutoWindowLevelFromRaw16(byte[] raw16LittleEndian, out int window, out int level)
        {
            int pixels = raw16LittleEndian.Length / 2;
            if (pixels <= 0)
            {
                window = 65535;
                level = 32768;
                return;
            }
    
            int min = 65535;
            int max = 0;
    
            for (int i = 0; i < pixels; i++)
            {
                int lo = raw16LittleEndian[i * 2 + 0];
                int hi = raw16LittleEndian[i * 2 + 1];
                int v = (hi << 8) | lo;
                if (v < min) min = v;
                if (v > max) max = v;
            }
    
            int w = max - min;
            if (w < 1) w = 1;
            int l = min + w / 2;
    
            window = Math.Max(1, Math.Min(65535, w));
            level  = Math.Max(0, Math.Min(65535, l));
        }

        private void SaveFrameIfEnabled(byte[] raw16Out, byte[] bgraForPng, int width, int height, int frameIndex)
        {
            if (CbSaveOn?.IsChecked != true) return;
            if (raw16Out == null || raw16Out.Length < width * height * 2) return;
            if (bgraForPng == null || bgraForPng.Length < width * height * 4) return;

            try
            {
                Directory.CreateDirectory(_outputDir);

                string stem = $"frame_{frameIndex:D6}";
                string rawPath = Path.Combine(_outputDir, stem + "_proc16.raw");
                string pngPath = Path.Combine(_outputDir, stem + "_view.png");

                File.WriteAllBytes(rawPath, raw16Out);
                SaveBgraToPng(bgraForPng, width, height, pngPath);
            }
            catch (Exception ex)
            {
                UpdateStatus($"Save failed: {ex.Message}");
            }
        }

        private static void SaveBgraToPng(byte[] bgra, int width, int height, string outPath)
        {
            var bs = BitmapSource.Create(width, height, 96, 96, PixelFormats.Bgra32, null, bgra, width * 4);
            var enc = new PngBitmapEncoder();
            enc.Frames.Add(BitmapFrame.Create(bs));
            using (var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.None))
                enc.Save(fs);
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

        private void SlWindow_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!_uiReady || _suppressParamEvents) return;
            _window = (int)Math.Round(SlWindow.Value);
            if (TbWindow != null) TbWindow.Text = _window.ToString();
            ApplyParamsToDll();
        }

        private void SlLevel_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (!_uiReady || _suppressParamEvents) return;
            _level = (int)Math.Round(SlLevel.Value);
            if (TbLevel != null) TbLevel.Text = _level.ToString();
            ApplyParamsToDll();
        }

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
                var p = NativeImageProc.MakeDefaultParams(
                    window: _window, level: _level,
                    enableEdge: _enableEdge,
                    enableBlur: _enableBlur,
                    enableInvert: _enableInvert,
                    enableThreshold: _enableThreshold,
                    thresholdValue: _thresholdValue);
                _processor.SetParameters(p);
            }
            catch (Exception ex)
            {
                UpdateStatus($"IPC_SetParams failed: {ex.Message}");
            }
        }

        private void UpdateStatus(string msg) => TbStatus.Text = msg;
        private void CbBlur_Checked(object sender, RoutedEventArgs e) { _enableBlur = 1; ApplyParamsToDll(); }
        private void CbBlur_Unchecked(object sender, RoutedEventArgs e) { _enableBlur = 0; ApplyParamsToDll(); }

        private void CbInvert_Checked(object sender, RoutedEventArgs e) { _enableInvert = 1; ApplyParamsToDll(); }
        private void CbInvert_Unchecked(object sender, RoutedEventArgs e) { _enableInvert = 0; ApplyParamsToDll(); }

        private void CbThreshold_Checked(object sender, RoutedEventArgs e) { _enableThreshold = 1; ApplyParamsToDll(); }
        private void CbThreshold_Unchecked(object sender, RoutedEventArgs e) { _enableThreshold = 0; ApplyParamsToDll(); }

        private void TbThresh_LostFocus(object sender, RoutedEventArgs e)
        {
            if (!int.TryParse(TbThresh.Text, out int v)) v = _thresholdValue;
            v = Math.Max(0, Math.Min(65535, v));
            _thresholdValue = v;
            TbThresh.Text = _thresholdValue.ToString();
            ApplyParamsToDll();
        }
    }
}
