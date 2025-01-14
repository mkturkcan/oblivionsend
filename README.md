<p align="center">
  <img src="https://github.com/mkturkcan/oblivionsend/blob/main/assets/logo.png?raw=true"  width="180" />
</p>


# Oblivion's End

A merged LoRA for gemma-2-9b-it, trained using DPO datasets for creative writing using [my DPO training notebook](https://github.com/mkturkcan/dpo-model-trainer).

## Model Details

### How to Use

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "mehmetkeremturkcan/oblivionsend",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

from transformers import TextStreamer
FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer)

inputs = tokenizer(
[
"""<start_of_turn>user
Write a story with the following description: Setting - a dark abandoned watchtower and its environs. A wizard carefully explores a tomb where a priest of a dark, dead God has raised a band of brigands that have been terrorizing a town."""+ """<end_of_turn>
<start_of_turn>model
"""
], return_tensors = "pt").to("cuda")

_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 4096, num_beams=1, temperature=1.0, do_sample=True)
```

### Model Description

- **Finetuned from model:** google/gemma-2-9b-it
### Model Sources [optional]

- **Repository:** [GitHub](https://github.com/mkturkcan/dpo-model-trainer/tree/main).

## Uses

Made for creative writing. 

## Training Details

### Training Data

Check out the model card details at HuggingFace.

### Training Procedure

Model training performance (margins) are available in the [wandb instance](https://api.wandb.ai/links/mkturkcan/4djkmhwp).

#### Training Hyperparameters

- **Training regime:** bf16 on a 1x 80GB A100 node.

## Environmental Impact

Total emissions are estimated to be 0.83 kgCO$_2$eq.
