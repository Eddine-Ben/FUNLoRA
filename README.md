# SegFormer Cityscapes FT — LoRA + Fisher-Guided Unfreezing (FUN-LoRA)

Fine-tune **SegFormer (MiT backbone)** on **Cityscapes (19 classes)** with:

- ✅ Auto-download of ADE20K pretrain weights (mmseg checkpoints)
- ✅ Two-phase training (**Head warmup → End-to-End**)
- ✅ Optional **LoRA** adapters on the backbone (PEFT or safe fallback)
- ✅ Optional **Fisher-guided selective unfreezing** (“FUN”)
- ✅ AMP, ReduceLROnPlateau, gradient accumulation
- ✅ Checkpoints + resume
- ✅ Memory helpers: resize+crop, gradient checkpointing, OOM recovery, `torch.compile` (where available)

---

## Repository layout


---

## Quickstart

### 1) Environment (Python ≥ 3.9; CUDA GPU recommended)

Install PyTorch to match your CUDA, then the remaining deps:

```bash
pip install torch torchvision   # choose versions per https://pytorch.org
pip install mmsegmentation opencv-python tqdm requests numpy
# Optional (native LoRA injection):
pip install peft

/path/to/cityscapes
├── leftImg8bit/{train,val,test}/..._leftImg8bit.png
└── gtFine/{train,val}/..._gtFine_labelTrainIds.png

python run.py \
  --dataset-root /path/to/cityscapes \
  --model-variant B3 \
  --batch-size 1 \
  --resize-short 1024 \
  --crop-size 768 1536 \
  --use-lora \
  --use-fun-unfreeze


--use-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --lora-lr 5e-5

--use-fun-unfreeze
--fisher-compute-interval 3
--fisher-batches 12
--max-groups-per-unfreeze 1


python run.py --dataset-root /data/cityscapes --model-variant B3 \
  --batch-size 1 --resize-short 1024 --crop-size 768 1536 \
  --use-lora --use-fun-unfreeze


python run.py --dataset-root /data/cityscapes --model-variant B5_HQ \
  --batch-size 1 --resize-short 1024 --crop-size 768 1536


python run.py --dataset-root /data/cityscapes --model-variant B2 \
  --freeze-backbone-epochs 5 --use-compile False

python run.py --dataset-root /data/cityscapes \
  --resume checkpoint_epoch_40.pth

| Flag                                                           | Meaning                                              |
| -------------------------------------------------------------- | ---------------------------------------------------- |
| `--model-variant {B0…B5,B5_FAST,B5_2X2,B5_HQ}`                 | Select MiT depth/width; the “B5_*” aliases map to B5 |
| `--dataset-root PATH`                                          | Cityscapes root (expects `leftImg8bit/` + `gtFine/`) |
| `--batch-size INT`                                             | Often 1–2 with large crops / single GPU              |
| `--resize-short INT`                                           | Resize short side before cropping (keeps aspect)     |
| `--crop-size H W`                                              | Train/val crop; val can use `--center-crop`          |
| `--use-lora`                                                   | Enable LoRA adapters on the backbone                 |
| `--use-fun-unfreeze`                                           | Turn on Fisher-guided selective unfreezing           |
| `--freeze-backbone-epochs INT`                                 | Head warmup duration (default 10)                    |
| `--head-lr / --lora-lr / --backbone-lr / --backbone-lr-thawed` | LR hierarchy (head ≥ lora > backbone)                |
| `--use-amp`                                                    | Mixed precision                                      |
| `--use-compile`                                                | `torch.compile` (disabled on Windows automatically)  |
| `--checkpoint-interval INT`                                    | Save periodic training state                         |
| `--resume PATH`                                                | Resume from `checkpoint_epoch_*.pth`                 |
