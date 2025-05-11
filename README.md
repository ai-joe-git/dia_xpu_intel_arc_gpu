# Dia XPU - Intel Arc GPU Support

This is a fork of the [Nari Labs Dia](https://github.com/nari-labs/dia) project with added support for Intel GPUs, specifically optimized for Intel Arc and Intel Core Ultra integrated graphics.

Dia is a 1.6B parameter text-to-speech model that directly generates highly realistic dialogue from transcripts. This fork enables you to run the model on Intel GPUs using the XPU backend in PyTorch.

## Intel GPU Support

This fork has been specifically modified to support:
- Intel Arc discrete GPUs
- Intel Core Ultra processors with integrated Arc graphics
- PyTorch XPU backend with Intel Extension for PyTorch

## Requirements

- Intel GPU drivers (version 30.0.100.9955 or later recommended)
- PyTorch nightly build with XPU support
- Intel Extension for PyTorch

## Setup and Installation

### Quick Setup with Intel XPU Support

```
# Clone this repository
git clone https://github.com/ai-joe-git/dia_xpu_intel_arc_gpu.git
cd dia_xpu_intel_arc_gpu

# Run the setup script
python setup_xpu.py

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate.bat  # On Windows

# Run the Gradio UI
python app.py --device xpu
```

### Docker Support

```
# Build the Docker image
docker build . -f docker/Dockerfile.gpu -t dia-xpu

# Run the container with Intel GPU access
docker run --rm --device=/dev/dri -p 7860:7860 dia-xpu
```

## Usage

### Python API

```
from dia.model import Dia

# Load model with Intel XPU support
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device="xpu")

# Generate audio from text
text = "[S1] Dia is running on Intel Arc GPU. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs)"
output = model.generate(text, verbose=True)

# Save the generated audio
model.save_audio("output.wav", output)
```

## Performance on Intel GPUs

| Precision | Realtime Factor | VRAM Usage |
|:---------:|:---------------:|:----------:|
| `bfloat16` | ~1.8x | ~8GB |
| `float16` | ~1.9x | ~8GB |
| `float32` | ~0.8x | ~12GB |

## Troubleshooting

If you encounter issues with Intel GPU detection:

1. Ensure you have the latest Intel GPU drivers installed
2. For Linux users, verify OpenCL runtime packages are installed:
   ```
   sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
   ```
3. For Windows users, check Device Manager to verify your Intel GPU is properly recognized

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Original Dia model by [Nari Labs](https://github.com/nari-labs/dia)
- Intel Extension for PyTorch team for XPU backend support
