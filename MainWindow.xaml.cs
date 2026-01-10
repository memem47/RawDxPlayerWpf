using System;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Threading;
using Microsoft.Win32;
using RawDxPlayerWpf.Dx;
using RawDxPlayerWpf.Processing;
using RawDxPlayerWpf.Raw;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RawDxPlayerWpf
{
    public partial class MainWindow : Window
    {
        private readonly DispatcherTimer _timer = new DispatcherTimer();
        private RawSequence _seq;
        private int _index;

        private DxRenderer _renderer;
        private D3DImageHost _d3dImageHost;

        // 将来ここを CUDA DLL 呼び出し実装に差し替える（今はnull=何もしない）
        private IImageProcessor _processor = null;
        private WriteableBitmap _wb;

        public MainWindow()
        {
            InitializeComponent();

            _timer.Tick += (_, __) => RenderNextFrame();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            UpdateStatus("Ready");
        }


        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            StopPlayback();
            _renderer?.Dispose();
            _d3dImageHost?.Dispose();
        }

        private void BtnOpen_Click(object sender, RoutedEventArgs e)
        {
            var lastPath = Properties.Settings.Default.LastOpenedFile;
            string initDir = null;
            string initFile = null;

            if (!string.IsNullOrWhiteSpace(lastPath) && File.Exists(lastPath))
            {
                initDir = Path.GetDirectoryName(lastPath);
                initFile = Path.GetFileName(lastPath);
            }

            var dlg = new OpenFileDialog
            {
                Title = "RAWフォルダ内の任意のRAW/BINファイルを選択してください",
                Filter = "RAW/BIN files (*.raw;*.bin)|*.raw;*.bin",
                CheckFileExists = true,
                InitialDirectory = initDir,
                FileName = initFile ?? ""
            };

            if (dlg.ShowDialog() != true) return;

            Properties.Settings.Default.LastOpenedFile = dlg.FileName;
            Properties.Settings.Default.Save();

            LoadSequenceFromFile(dlg.FileName);
        }

        private void LoadSequenceFromFile(string filePath)
        {
            StopPlayback();

            _seq = RawSequence.FromAnyFileInFolder(filePath);
            _index = 0;

            // Rendererを作り直す（サイズ変わる可能性があるため）
            _renderer?.Dispose();
            _renderer = new DxRenderer(_seq.Width, _seq.Height);

            // WPF表示用の WriteableBitmap を作成
            _wb = new WriteableBitmap(
                _seq.Width, _seq.Height,
                96, 96,
                PixelFormats.Bgra32,
                null);

            ImgView.Source = _wb;

            UpdateStatus($"Loaded: {_seq.Files.Count} frames, {_seq.Width}x{_seq.Height}, folder={_seq.Folder}");

            // 1枚目表示
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

            var path = _seq.Files[_index];

            // Window/Level（簡易：16bit→8bitに落として表示）
            int window = ParseOrDefault(TbWindow.Text, 4000);
            int level = ParseOrDefault(TbLevel.Text, 2000);

            // 読み込み（16-bit）→ BGRA(8-bit) へ変換
            var bgra = RawFrameReader.Load16ToBgra8(path, _seq.Width, _seq.Height, window, level);

            // 将来：ここで _processor.Process(in/out texture) に切り替える
            // 今はCPU変換結果をD3D11テクスチャへ更新
            _renderer.UpdateFrame(bgra);


            // 画面表示（WriteableBitmap）
            _wb.WritePixels(
                new Int32Rect(0, 0, _seq.Width, _seq.Height),
                bgra,
                _seq.Width * 4,   // stride
                0);


            TbStatus.Text = $"Frame {_index + 1}/{_seq.Files.Count} : {System.IO.Path.GetFileName(path)}";
        }

        private static int ParseOrDefault(string s, int def)
            => int.TryParse(s, out var v) ? v : def;

        private void UpdateStatus(string msg) => TbStatus.Text = msg;
    }
}
