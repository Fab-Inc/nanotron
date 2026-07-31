# %%
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]

# %%
checkpoint = "HuggingFaceTB/SmolLM3-3B-Base"
# revision = "stage3-step-4720000"  # replace by the revision you want
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if hasattr(torch, "mps") and torch.mps.is_available() else "cpu"
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)#, revision=revision)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)#, revision=revision).to(device)
inputs = tokenizer.encode("Gravity is", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

model_checkpoint_dir = ROOT / "checkpoints" / "smollm3-checkpoint"
model.save_pretrained(str(model_checkpoint_dir))
tokenizer.save_pretrained(str(model_checkpoint_dir))
