# tests/test_quantization.py
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent dir to path
from atlasinfer_core.quant.adaptive_quantizer import quantize_tensor, dequantize_tensor, QuantizedTensor, clear_dequantized_cache

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test Case 1: Standard Tensor
    print("\n--- Test Case 1: Standard ---")
    original_tensor = (torch.randn(128, 256, dtype=torch.float32) * 5).to(torch.float16).to(device)
    original_tensor.view(-1)[::150] *= 10 # Add outliers

    print(f"Original shape: {original_tensor.shape}, dtype: {original_tensor.dtype}")
    original_mem = original_tensor.element_size() * original_tensor.nelement()
    print(f"Original memory (MB): {original_mem / (1024**2):.3f}")

    # Quantize (move to CPU first if needed)
    quant_config = {"block_size": 64, "z_threshold": 3.5}
    quantized_data = quantize_tensor(original_tensor.cpu(), **quant_config)

    quant_mem = quantized_data.memory_footprint()
    print(f"Quantized memory (MB): {quant_mem / (1024**2):.3f}")
    print(f"Compression ratio: {original_mem / quant_mem:.2f}x")
    print(f"Num outliers: {quantized_data.fp16_outliers.numel()}")

    # Dequantize (move back to target device)
    clear_dequantized_cache() # Clear before test
    reconstructed_tensor = dequantize_tensor(quantized_data, target_device=device)
    # Test cache
    reconstructed_tensor_cached = dequantize_tensor(quantized_data, target_device=device)


    print(f"Reconstructed shape: {reconstructed_tensor.shape}, dtype: {reconstructed_tensor.dtype}")
    diff = torch.abs(original_tensor - reconstructed_tensor).float()
    mse = torch.mean(diff**2)
    print(f"Reconstruction MSE: {mse.item():.6f}")
    assert mse < 1e-3, "MSE too high!" # Basic check

    # Test Case 2: Small Tensor (smaller than block size)
    print("\n--- Test Case 2: Small Tensor ---")
    original_small = torch.randn(10, dtype=torch.float16).to(device)
    quant_small = quantize_tensor(original_small.cpu(), block_size=64)
    recon_small = dequantize_tensor(quant_small, target_device=device)
    mse_small = torch.mean((original_small - recon_small).float()**2)
    print(f"Small tensor MSE: {mse_small.item():.6f}")
    assert mse_small < 1e-3

    # Test Case 3: Empty Tensor
    print("\n--- Test Case 3: Empty Tensor ---")
    original_empty = torch.empty((0, 10), dtype=torch.float16).to(device)
    quant_empty = quantize_tensor(original_empty.cpu())
    recon_empty = dequantize_tensor(quant_empty, target_device=device)
    assert recon_empty.shape == original_empty.shape

    clear_dequantized_cache()
    print("\nQuantization tests passed!")