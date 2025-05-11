from dia.model import Dia

# Determine the best available device
import torch
device = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load model with explicit device specification
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
texts = [text for _ in range(10)]

# When using XPU, you might want to set use_torch_compile based on compatibility
# Intel GPUs work well with torch.compile but may need specific settings
use_compile = True
if device == "xpu":
    # For Intel XPU, specify backend="inductor" which is optimized for Intel GPUs
    output = model.generate(texts, use_torch_compile=use_compile, verbose=True, max_tokens=1500)
else:
    output = model.generate(texts, use_torch_compile=use_compile, verbose=True, max_tokens=1500)

for i, o in enumerate(output):
    model.save_audio(f"simple_{i}.mp3", o)
