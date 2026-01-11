using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RawDxPlayerWpf.Raw
{
    public sealed class RawSequence
    {
        public string Folder { get; }
        public List<string> Files { get; }
        public int Width { get; }
        public int Height { get; }

        private RawSequence(string folder, List<string> files, int size)
        {
            Folder = folder;
            Files = files;
            Width = size;
            Height = size;
        }

        public static RawSequence FromAnyFileInFolder(string selectedFile)
        {
            var folder = Path.GetDirectoryName(selectedFile);

            // 同フォルダの .raw/.bin を連番として扱う（名前でソート）
            var files = Directory.GetFiles(folder, "*.*")
                .Where(p =>
                {
                    var ext = Path.GetExtension(p).ToLowerInvariant();
                    return ext == ".raw" || ext == ".bin";
                })
                .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (files.Count == 0)
                throw new InvalidOperationException("No .raw/.bin files found.");

            // サイズ推定は「選択されたファイル」を基準にする
            int size = GuessSquareSizeFromFile(selectedFile);
            return new RawSequence(folder, files, size);
        }

        private static int GuessSquareSizeFromFile(string path)
        {
            long bytes = new FileInfo(path).Length;
            if (bytes <= 0 || (bytes % 2) != 0)
                throw new InvalidOperationException("File size is not valid for 16-bit grayscale.");

            long pixels = bytes / 2;
            double root = Math.Sqrt(pixels);
            int n = (int)Math.Round(root);

            // 完全正方形なら採用
            if ((long)n * n == pixels)
                return n;

            // よくある候補（必要に応じて増やしてOK）
            int[] candidates = new[] { 2048, 1536, 1024, 768, 640, 512, 384, 256 };

            // 「n*nに最も近い候補」を採用
            int best = candidates
                .OrderBy(c => Math.Abs((long)c * c - pixels))
                .First();

            return best;
        }
    }
}
