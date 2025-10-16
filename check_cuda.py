import torch

print("="*60)
print("PYTORCH CUDA CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("\n⚠️  CUDA NOT AVAILABLE - Training will be VERY SLOW on CPU")
    print("\nTo fix this, install PyTorch with CUDA:")
    print("  pip uninstall torch")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
print("="*60)
