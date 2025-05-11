from dia.model import Dia
import torch

# Determine the best available device
device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load model with explicit device specification
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

# Try to import Intel Extension for PyTorch for additional optimizations
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# Configure torch.compile options for Intel GPU
if device == "xpu" and has_ipex:
    # Use inductor backend which is optimized for Intel GPUs
    output = model.generate(text, use_torch_compile=True, verbose=True)
else:
    output = model.generate(text, use_torch_compile=True, verbose=True)

model.save_audio("simple.mp3", output)
