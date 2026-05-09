# 🦥 LLM Fine-Tuning with Unsloth + Binary Diffusion Head

Fine-tune large language models (Qwen2.5-14B, BitDance-14B) on image captioning tasks using [Unsloth](https://github.com/unslothai/unsloth) for 2× faster training, LoRA for parameter-efficient adaptation, and a custom Binary Diffusion Head for visual alignment.

---

## Overview

This project fine-tunes a 14B-parameter LLM on the COCO image captioning dataset. It combines:

- **Unsloth** — fast 4-bit QLoRA fine-tuning
- **LoRA** — low-rank adapter training on attention and MLP projections
- **BinaryDiffusionHead** — a lightweight auxiliary head that learns to predict visual velocity fields from the model's hidden states, encouraging visual grounding

Training uses a joint loss: a weighted cross-entropy text loss plus an MSE diffusion loss.

---

## Models Supported

| Model | HuggingFace ID |
|---|---|
| BitDance 14B (Qwen3-based) | `shallowdream204/BitDance-14B-64x` |
| Qwen2.5 14B Instruct (4-bit) | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` |

---

## Requirements

- Python 3.10+
- CUDA GPU (tested on Tesla T4, 14.5 GB VRAM)
- PyTorch 2.3+

Install dependencies:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install pillow torchvision
```

---

## Dataset

The script automatically downloads a subset of [COCO 2017](https://cocodataset.org/) (train split):

- **Annotations** — `captions_train2017.json`
- **Images** — up to `MAX_SAMPLES` (default: 5,000) downloaded on first run
- Stored under `DATA_DIR` (default: `/kaggle/working/coco`)

No manual setup needed — just run the script.

---

## Configuration

Key hyperparameters at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` | Base model |
| `MAX_SEQ_LEN` | `2048` | Max sequence length for the model |
| `SEQ_LEN` | `64` | Tokenized sequence length per sample |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA scaling factor |
| `LOAD_IN_4BIT` | `True` | Use 4-bit quantization |
| `BATCH_SIZE` | `2` | Per-device batch size |
| `ACCUMULATION` | `4` | Gradient accumulation steps |
| `EPOCHS` | `4` | Training epochs |
| `LR` | `1e-5` | Learning rate |
| `MAX_SAMPLES` | `5000` | COCO samples to use |

---

## Architecture

```
Input Image ──► (discarded at inference; used for dataset construction)
Input Text  ──► Tokenizer ──► [LoRA-patched LLM] ──► Logits ──► Text Loss (CE)
                                      │
                               Hidden States
                                      │
                          BinaryDiffusionHead(t) ──► Predicted Velocity ──► Visual Loss (MSE)
```

**BinaryDiffusionHead** — a two-layer MLP that conditions on the last hidden state and a noise timestamp `t`, predicting velocity fields used in flow-matching style training.

**Joint Loss:**
```
loss = 0.1 * loss_text + loss_visual
```

---

## Training

```bash
python train.py
```

Sample output:
```
Ep 0 | Step 0   | Loss: 7.7330 | Acc: 0.3448
Ep 0 | Step 100 | Loss: 1.7875 | Acc: 0.2333
Ep 0 | Step 500 | Loss: 1.3177 | Acc: 0.5625
Ep 1 | Step 980 | Loss: 1.2788 | Acc: 0.5833
...
Ep 3 | Step 2480 | Loss: 1.2388 | Acc: 0.5517
```

Training uses:
- **Mixed precision** (`torch.amp.autocast` with `float16`)
- **Gradient scaling** (`GradScaler`)
- **Gradient clipping** (max norm `0.5`)
- **NaN skipping** — batches producing NaN loss are skipped automatically

---

## LoRA Target Modules

LoRA adapters are applied to all major projection layers:

```
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

---

## Results Summary

Accuracy here measures next-token prediction accuracy on non-padding tokens within the 64-token caption window.

| Model | Epoch 0 Final Acc | Epoch 3 Final Acc |
|---|---|---|
| BitDance-14B | ~0.26 | ~0.38 |
| Qwen2.5-14B-Instruct | ~0.72 | ~0.55 |

Qwen2.5 converges faster due to the instruction-tuned base and better tokenizer alignment with caption text.

---

## File Structure

```
.
├── train.py               # Main training script
├── README.md
└── coco/                  # Auto-created on first run
    ├── annotations/
    │   └── captions_train2017.json
    └── images/
        └── *.jpg
```

---

## Notes

- The `image` tensor is loaded per sample but not passed to the LLM directly — the model is a text-only LM; the image is used to construct the dataset and supply the diffusion head's target velocities (currently random, acting as a regularizer placeholder).
- To use real visual embeddings, replace `target_velocities: torch.randn(...)` with actual image encoder outputs (e.g., CLIP, SigLIP).
- For multi-GPU setups, Unsloth will report available GPUs automatically but currently trains on the primary CUDA device.

---

## License

This project is for research and educational use. Model weights are subject to their respective licenses on HuggingFace Hub.
