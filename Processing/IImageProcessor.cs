namespace RawDxPlayerWpf.Processing
{
    // ②で C++/CUDA DLL を呼ぶときの抽象化
    // 今回は未使用（null）でOK。
    public interface IImageProcessor
    {
        void Initialize(int gpuId, System.IntPtr inputDxSharedHandle, System.IntPtr outputDxSharedHandle);
        void SetParameters(object paramStruct);
        void Execute();
        void Shutdown();
    }
}
