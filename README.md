# Dia TTS - CPU Mode (Intel Arc GPU Support Discontinued)

## Project Status Update

After extensive testing and troubleshooting, we have reverted this project to CPU-only mode due to significant challenges with Intel Arc GPUs for text-to-speech workloads. This README explains our findings and why we made this decision.

## Why We Discontinued Intel Arc GPU Support

### 1. Audio Distortion Issues

The primary reason for reverting to CPU-only mode was persistent audio distortion when using Intel Arc GPUs. Despite multiple approaches to resolve this issue:

- We implemented a separate CPU-only audio decoding pipeline while keeping token generation on the GPU
- We tried various data type configurations (float32, float16, bfloat16)
- We added explicit tensor detachment and careful memory management
- We isolated the DAC (Descript Audio Codec) model on CPU

All of these approaches still resulted in distorted, unusable audio output when the model utilized the Intel Arc GPU for any part of the processing pipeline.

### 2. Performance Considerations

Contrary to our initial expectations, our testing revealed that Intel Arc GPUs do not provide significant performance benefits for text-to-speech workloads compared to CPU:

- Our logs showed a realtime factor of only ~0.08x when using the Intel Arc GPU
- Similar findings were reported by other users, with one noting that Speech T5 on Intel Arc GPU 770 took 8 seconds compared to 3 seconds on CPU
- The overhead of data transfer between CPU and GPU negated any computational advantages

### 3. Compatibility Challenges

We encountered several compatibility issues when trying to optimize for Intel Arc GPUs:

- Conflicts between Intel Extension for PyTorch (IPEX) and other PyTorch components
- Triton backend errors when attempting to use torch.compile with XPU devices
- Gradient tracking issues during audio processing
- Type conversion problems between the model's internal representation and the audio codec

### 4. Known Intel Arc GPU Audio Processing Limitations

Our research uncovered that Intel Arc GPUs have documented issues with audio processing:

- Audio distortion, popping, and stuttering have been reported by multiple users
- Some users reported that audio stops working after a few seconds
- Others noted that audio requires device manager resets to function properly
- These issues appear to be related to the GPU's audio processing pipeline rather than specific to our implementation

## Current Implementation

The current implementation runs entirely on CPU, which provides:

1. Reliable, high-quality audio output without distortion
2. Consistent performance without unexpected errors
3. Broader compatibility across different systems
4. Simpler codebase without the need for complex device management

## Technical Details

Our implementation journey included:

1. Initially implementing the model with XPU support using PyTorch's native capabilities
2. Adding Intel Extension for PyTorch (IPEX) optimizations
3. Creating a hybrid approach with token generation on GPU and audio processing on CPU
4. Implementing careful memory management and tensor type handling
5. Finally, reverting to a fully CPU-based implementation for reliability

## Future Considerations

While Intel Arc GPUs show promise for many AI workloads, our experience suggests they are not yet optimal for text-to-speech applications, particularly those involving complex audio processing pipelines like the Dia model. As Intel's drivers and software stack mature, we may revisit GPU acceleration in the future.

## Acknowledgements

- Original Dia model by [Nari Labs](https://github.com/nari-labs/dia)
- The PyTorch and Intel teams for their ongoing work to improve GPU support

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.