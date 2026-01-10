using System;

namespace RawDxPlayerWpf.Processing
{
    public sealed class NativeImageProcessor : IImageProcessor
    {
        private bool _initialized;

        public void Initialize(int gpuId, IntPtr inputDxSharedHandle, IntPtr outputDxSharedHandle)
        {
            int r = NativeImageProc.IPC_Init(gpuId, inputDxSharedHandle, outputDxSharedHandle);
            if (r != 0) throw new InvalidOperationException($"IPC_Init failed: {r}");
            _initialized = true;
        }

        public void SetParameters(object paramStruct)
        {
            if (!_initialized) return;

            if (paramStruct is NativeImageProc.IPC_Params p)
            {
                int r = NativeImageProc.IPC_SetParams(ref p);
                if (r != 0) throw new InvalidOperationException($"IPC_SetParams failed: {r}");
                return;
            }

            throw new ArgumentException("paramStruct must be NativeImageProc.IPC_Params");
        }

        public void Execute()
        {
            if (!_initialized) return;
            int r = NativeImageProc.IPC_Execute();
            if (r != 0) throw new InvalidOperationException($"IPC_Execute failed: {r}");
        }

        public void Shutdown()
        {
            if (!_initialized) return;
            NativeImageProc.IPC_Shutdown();
            _initialized = false;
        }
    }
}
