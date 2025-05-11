# Dia XPU - Intel Arc GPU Support

This is a fork of the [Nari Labs Dia](https://github.com/nari-labs/dia) project with added support for Intel GPUs, specifically optimized for Intel Arc and Intel Core Ultra integrated graphics.

## Overview

Dia is a 1.6B parameter text-to-speech model that directly generates highly realistic dialogue from transcripts. This fork enables you to run the model on Intel GPUs using the XPU backend in PyTorch.

## Features

- **Intel GPU Support**: Run Dia on Intel Arc discrete GPUs and Intel Core Ultra integrated graphics
- **PyTorch XPU Backend**: Utilizes the latest PyTorch nightly builds with XPU support
- **Intel Extension for PyTorch**: Optimized performance with Intel-specific acceleration
- **Voice Cloning**: Condition the output on audio samples to control emotion and tone
- **Nonverbal Communication**: Generate realistic nonverbal elements like laughter, coughing, etc.

## Quick Start with Batch File

The easiest way to get started is to use the provided batch file:

1. Download the `run_dia_xpu.bat` file from this repository
2. Place it in a folder where you want to install the project
3. Double-click the batch file to automatically:
   - Clone the repository
   - Set up the virtual environment with Intel XPU support
   - Launch the Gradio UI with Intel XPU device selected

The batch file includes checks to avoid re-cloning or re-setting up if you've already done those steps, so it's safe to run multiple times.

### Batch File Contents

For reference, here's what the batch file contains:

```batch
@echo off
echo Setting up Dia XPU for Intel Arc GPU...

REM Clone the repository if it doesn't exist
if not exist dia_xpu_intel_arc_gpu (
    echo Cloning repository...
    git clone https://github.com/ai-joe-git/dia_xpu_intel_arc_gpu.git
    cd dia_xpu_intel_arc_gpu
) else (
    echo Repository already exists, using existing folder
    cd dia_xpu_intel_arc_gpu
)

REM Run the setup script if venv doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python setup_xpu.py
) else (
    echo Virtual environment already exists
)

REM Activate the virtual environment and run the app
echo Activating virtual environment and starting Gradio UI...
call venv\Scripts\activate.bat
python app.py --device xpu

pause
```

## Manual Installation

If you prefer to install manually:

```bash
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

## System Requirements

- Intel Arc GPU or Intel Core Ultra processor with integrated graphics
- Intel GPU drivers (version 30.0.100.9955 or later recommended)
- Python 3.10 or newer
- Windows 10/11 or Linux with Intel GPU support

## Using Existing Model Files

If you already have the Dia model files downloaded from a previous installation (e.g., from Pinokio), you can reuse them to avoid downloading again. The model will automatically look for cached files in the standard Hugging Face cache location.

## Performance

Performance on Intel Core Ultra processors with integrated Arc graphics:

| Precision | Realtime Factor w/ compile | Realtime Factor w/o compile | VRAM |
|:---------:|:-------------------------:|:---------------------------:|:----:|
| `bfloat16` | ~1.8x | ~1.2x | ~8GB |
| `float16` | ~1.9x | ~1.1x | ~8GB |
| `float32` | ~0.8x | ~0.7x | ~12GB |

## Python API Examples

### Basic Generation

```python
from dia.model import Dia
import torch

# Use Intel XPU if available
device = "xpu" if torch.xpu.is_available() else "cpu"
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

text = "[S1] Dia is running on Intel Arc GPU. [S2] You get full control over scripts and voices."
output = model.generate(text, verbose=True)

model.save_audio("output.wav", output)
```

### Voice Cloning

```python
from dia.model import Dia
import torch

device = "xpu" if torch.xpu.is_available() else "cpu"
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

# Transcript of the voice you want to clone
clone_from_text = "[S1] This is the voice I want to clone."
clone_from_audio = "voice_sample.mp3"  # Your audio file

# Text to generate with the cloned voice
text_to_generate = "[S1] Hello, this is the cloned voice speaking."

# Generate with voice cloning
output = model.generate(
    clone_from_text + text_to_generate, 
    audio_prompt=clone_from_audio, 
    use_torch_compile=True, 
    verbose=True
)

model.save_audio("cloned_voice.mp3", output)
```

## Troubleshooting

If you encounter issues with Intel GPU detection:

1. Ensure you have the latest Intel GPU drivers installed
2. For Linux users, verify OpenCL runtime packages are installed:
   ```
   sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
   ```
3. For Windows users, check Device Manager to verify your Intel GPU is properly recognized
4. Make sure you're using the PyTorch nightly build with XPU support

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Original Dia model by [Nari Labs](https://github.com/nari-labs/dia)
- Intel Extension for PyTorch team for XPU backend support