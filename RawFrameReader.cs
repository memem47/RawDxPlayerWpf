using System;
using System.IO;

namespace RawDxPlayerWpf.Raw
{
    public static class RawFrameReader
    {
        // window/level: 16bit値を [level-window/2, level+window/2] にクリップして 0..255 にマップ
        public static byte[] Load16ToBgra8(string path, int width, int height, int window, int level)
        {
            int pixels = checked(width * height);
            int bytesNeeded = checked(pixels * 2);

            byte[] raw = File.ReadAllBytes(path);
            if (raw.Length < bytesNeeded)
                throw new InvalidOperationException($"File too small: {path}");

            byte[] bgra = new byte[pixels * 4];

            int min = level - (window / 2);
            int max = level + (window / 2);
            if (max <= min) { min = 0; max = 65535; }

            for (int i = 0; i < pixels; i++)
            {
                // little-endian 16-bit
                int lo = raw[i * 2 + 0];
                int hi = raw[i * 2 + 1];
                int v16 = (hi << 8) | lo;

                int v = v16;
                if (v < min) v = min;
                if (v > max) v = max;

                int v8 = (int)((v - min) * 255.0 / (max - min));
                byte b = (byte)(v8 < 0 ? 0 : (v8 > 255 ? 255 : v8));

                // BGRA
                int o = i * 4;
                bgra[o + 0] = b;
                bgra[o + 1] = b;
                bgra[o + 2] = b;
                bgra[o + 3] = 255;
            }

            return bgra;
        }
    }
}
