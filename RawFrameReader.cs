using System;
using System.IO;

namespace RawDxPlayerWpf.Raw
{
    public static class RawFrameReader
    {
        /// <summary>
        /// Read RAW16 bytes (little endian) from file.
        /// Returns exactly width*height*2 bytes.
        /// </summary>
        public static byte[] Load16RawBytes(string path, int width, int height)
        {
            int bytesNeeded = checked(width * height * 2);
            byte[] raw = File.ReadAllBytes(path);

            if (raw.Length < bytesNeeded)
                throw new InvalidOperationException($"File too small: {path}");

            if (raw.Length == bytesNeeded) return raw;

            // Trim if file has extra bytes
            var trimmed = new byte[bytesNeeded];
            Buffer.BlockCopy(raw, 0, trimmed, 0, bytesNeeded);
            return trimmed;
        }

        /// <summary>
        /// CPU WL/WW: Convert RAW16 bytes to BGRA8 for display.
        /// window/level: clip [level-window/2, level+window/2] then map 0..255.
        /// </summary>
        public static byte[] Convert16ToBgra8(byte[] raw16LittleEndian, int width, int height, int window, int level)
        {
            int pixels = checked(width * height);
            int bytesNeeded = checked(pixels * 2);

            if (raw16LittleEndian == null) throw new ArgumentNullException(nameof(raw16LittleEndian));
            if (raw16LittleEndian.Length < bytesNeeded)
                throw new ArgumentException("raw16 buffer is too small", nameof(raw16LittleEndian));

            if (window < 1) window = 1;
            int min = level - window / 2;
            int max = level + window / 2;
            if (max <= min) max = min + 1;

            byte[] bgra = new byte[pixels * 4];

            for (int i = 0; i < pixels; i++)
            {
                int b0 = raw16LittleEndian[i * 2 + 0];
                int b1 = raw16LittleEndian[i * 2 + 1];
                int v = (b1 << 8) | b0; // little endian

                if (v < min) v = min;
                if (v > max) v = max;

                int v8 = (int)((v - min) * 255.0 / (max - min));
                if (v8 < 0) v8 = 0;
                if (v8 > 255) v8 = 255;

                byte g = (byte)v8;
                int o = i * 4;
                bgra[o + 0] = g;   // B
                bgra[o + 1] = g;   // G
                bgra[o + 2] = g;   // R
                bgra[o + 3] = 255; // A
            }

            return bgra;
        }

        /// <summary>
        /// Convenience: file -> RAW16 -> BGRA8 (CPU WL/WW).
        /// </summary>
        public static byte[] Load16ToBgra8(string path, int width, int height, int window, int level)
        {
            var raw16 = Load16RawBytes(path, width, height);
            return Convert16ToBgra8(raw16, width, height, window, level);
        }
    }
}
