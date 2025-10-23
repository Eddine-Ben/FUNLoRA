from typing import List, Tuple
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Optional PEFT (LoRA)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_OK = True
except Exception as _peft_err:
    _PEFT_OK = False
    _PEFT_ERR = _peft_err


# -------------------------
# Model creation (SegFormer-like with MiT backbone via mmseg)
# -------------------------
def create_model(backbone: str, out_channels: int, pretrained: bool = True) -> nn.Module:
    """
    Builds a SegFormer-style model with MixVisionTransformer backbone (mmseg).
    The lightweight decoder projects multi-scale features and fuses them.
    """
    try:
        from mmseg.models.backbones.mit import MixVisionTransformer
    except Exception as e:
        raise ImportError("mmsegmentation is required for the model builder") from e

    arch = backbone.lower()
    variant = backbone.upper()
    embed_dims_map = {
        'B0': [32, 64, 160, 256],
        'B1': [64, 128, 320, 512],
        'B2': [64, 128, 320, 512],
        'B3': [64, 128, 320, 512],
        'B4': [64, 128, 320, 512],
        'B5': [64, 128, 320, 512]
    }
    depths_map = {
        'b0': [2, 2, 2, 2],
        'b1': [2, 2, 2, 2],
        'b2': [3, 4, 6, 3],
        'b3': [3, 4, 18, 3],
        'b4': [3, 8, 27, 3],
        'b5': [3, 6, 40, 3]
    }
    embed_dims_list = embed_dims_map.get(variant, [64, 128, 320, 512])
    depths = depths_map.get(arch, [3, 6, 40, 3])
    num_heads = [1, 2, 5, 8]
    sr_ratios = [8, 4, 2, 1]

    class Net(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            try:
                self.backbone = MixVisionTransformer(
                    embed_dims=embed_dims_list, num_heads=num_heads, mlp_ratios=[4, 4, 4, 4],
                    qkv_bias=True, depths=depths, sr_ratios=sr_ratios,
                    drop_rate=0.0, drop_path_rate=0.1, out_indices=(0, 1, 2, 3)
                )
            except TypeError:
                # handle older mmseg API
                self.backbone = MixVisionTransformer(
                    embed_dims=embed_dims_list[0], num_layers=depths, num_heads=num_heads, mlp_ratio=4,
                    qkv_bias=True, sr_ratios=sr_ratios, out_indices=(0, 1, 2, 3),
                    drop_rate=0.0, drop_path_rate=0.1
                )
            cproj = 256
            self.proj = nn.ModuleList([nn.Conv2d(c, cproj, 1) for c in embed_dims_list])
            self.fuse = nn.Sequential(
                nn.Conv2d(cproj * 4, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.classifier = nn.Conv2d(256, num_classes, 1)

        def forward(self, x: th.Tensor) -> th.Tensor:
            feats = self.backbone(x)
            if isinstance(feats, dict):  # some mmseg versions return dict
                feats = [feats[k] for k in ['x1', 'x2', 'x3', 'x4'] if k in feats]
            h0, w0 = feats[0].shape[-2:]
            ups = []
            for i, f in enumerate(feats):
                p = self.proj[i](f)
                if p.shape[-2:] != (h0, w0):
                    p = F.interpolate(p, size=(h0, w0), mode='bilinear', align_corners=False)
                ups.append(p)
            y = th.cat(ups, dim=1)
            y = self.fuse(y)
            y = F.dropout(y, 0.1, training=self.training)
            y = self.classifier(y)
            return y

    return Net(out_channels)


# -------------------------
# LoRA helpers (PEFT-first, safe fallback)
# -------------------------
def _guess_lora_targets() -> List[str]:
    return ["qkv", "query", "key", "value", "proj", "fc1", "fc2", "mlp.fc1", "mlp.fc2", "out_proj"]


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear (safe fallback when PEFT isn't available)."""
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = int(r)
        self.scaling = (alpha / float(r)) if r and r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.r > 0:
            self.lora_A = nn.Linear(self.base.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.base(x)
        if self.r and self.r > 0:
            y = y + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return y


class LoRAConv1x1(nn.Module):
    """LoRA for 1×1 Conv2d (rare in backbone here, provided for completeness)."""
    def __init__(self, base: nn.Conv2d, r: int, alpha: int, dropout: float):
        super().__init__()
        assert base.kernel_size == (1, 1), "LoRAConv1x1 supports only 1×1 conv."
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.scaling = (alpha / float(r)) if r and r > 0 else 1.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        if self.r > 0:
            self.lora_A = nn.Conv2d(self.base.in_channels, self.r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(self.r, self.base.out_channels, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.base(x)
        if self.r and self.r > 0:
            y = y + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return y


def _wrap_module_with_lora(module: nn.Module, r: int, alpha: int, dropout: float):
    if isinstance(module, nn.Linear):
        return LoRALinear(module, r, alpha, dropout)
    if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
        return LoRAConv1x1(module, r, alpha, dropout)
    return None


def _inject_lora_modules(root: nn.Module, prefix: str, targets, r: int, alpha: int, dropout: float, verbose=True):
    n_wrapped, n_skipped = 0, 0
    for name, child in list(root.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if any(t in full.lower() for t in targets):
            wrapped = _wrap_module_with_lora(child, r, alpha, dropout)
            if wrapped is not None:
                setattr(root, name, wrapped)
                n_wrapped += 1
                if verbose:
                    print(f"🔧 LoRA injected -> {full} ({child.__class__.__name__})")
                continue
            else:
                n_skipped += 1
        w, s = _inject_lora_modules(child, full, targets, r, alpha, dropout, verbose)
        n_wrapped += w
        n_skipped += s
    return n_wrapped, n_skipped


def add_lora_to_backbone(model: nn.Module, r=8, alpha=16, dropout=0.05) -> nn.Module:
    targets = [t.lower() for t in _guess_lora_targets()]

    if _PEFT_OK:
        try:
            cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                target_modules=targets
            )
            if hasattr(model, "backbone"):
                model.backbone = get_peft_model(model.backbone, cfg)
            else:
                model = get_peft_model(model, cfg)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"✨ LoRA injected (PEFT): trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
            return model
        except Exception as e:
            print(f"⚠️ PEFT injection failed: {e} • falling back to built-in LoRA.")

    sub = model.backbone if hasattr(model, "backbone") else model
    wrapped, skipped = _inject_lora_modules(
        sub, "backbone" if hasattr(model, "backbone") else "", targets, r, alpha, dropout, verbose=True
    )
    # Freeze base weights inside wrappers
    for n, p in model.named_parameters():
        if ".base." in n:
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✨ LoRA injected (fallback): wrapped={wrapped}, skipped={skipped} "
          f"• trainable {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    return model


# -------------------------
# Training utils around model param selection
# -------------------------
def is_head_param(name: str) -> bool:
    lname = name.lower()
    return any(k in lname for k in ["classifier", "cls", "fuse", "proj", "decode", "head"])


def is_lora_param(name: str) -> bool:
    return "lora_" in name or ".lora_" in name


def param_groups(model: nn.Module, lr_head: float, lr_lora: float, lr_backbone: float,
                 wd_head: float = 1e-4, wd_lora: float = 0.0, wd_backbone: float = 1e-5):
    head, lora, backbone = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_lora_param(n):
            lora.append(p)
        elif is_head_param(n):
            head.append(p)
        else:
            backbone.append(p)
    groups = []
    if head: groups.append({"params": head, "lr": lr_head, "weight_decay": wd_head, "name": "head"})
    if lora: groups.append({"params": lora, "lr": lr_lora, "weight_decay": wd_lora, "name": "lora"})
    if backbone: groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": wd_backbone, "name": "backbone"})
    return groups


def set_requires_grad_named(model: nn.Module, pred, flag: bool):
    for n, p in model.named_parameters():
        if pred(n):
            p.requires_grad = flag


def group_frozen_backbone_modules(model: nn.Module):
    """
    Return { module_name: {'module': module, 'params': [frozen_params]} } for backbone-ish groups.
    """
    groups = {}
    for name, module in model.named_modules():
        if is_head_param(name):  # skip head
            continue
        lname = name.lower()
        if any(k in lname for k in ["stage", "block", "layer", "encoder"]):
            params = [p for _, p in module.named_parameters(recurse=False) if not p.requires_grad]
            if params:
                groups[name] = {"module": module, "params": params}
    return groups


def enable_mit_gradient_checkpointing(model: nn.Module):
    """
    Enable 'with_cp' (checkpointing) on MiT blocks to reduce memory.
    Has no effect on modules that don't have this attribute.
    """
    toggled = 0
    for m in model.modules():
        if hasattr(m, "with_cp"):
            try:
                m.with_cp = True
                toggled += 1
            except Exception:
                pass
    if toggled:
        print(f"🧠 Gradient checkpointing enabled on {toggled} modules.")
