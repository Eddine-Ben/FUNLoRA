from typing import Dict

# Local model variant registry + local file names
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "B0": {"weight_file": "weights/segformer_mit_b0_ade20k.pth"},
    "B1": {"weight_file": "weights/segformer_mit_b1_ade20k.pth"},
    "B2": {"weight_file": "weights/segformer_mit_b2_ade20k.pth"},
    "B3": {"weight_file": "weights/segformer_mit_b3_ade20k.pth"},
    "B4": {"weight_file": "weights/segformer_mit_b4_ade20k.pth"},
    "B5": {"weight_file": "weights/segformer_mit_b5_ade20k.pth"},
    # “++” aliases map to B5 weights
    "B5_FAST": {"weight_file": "weights/segformer_mit_b5_ade20k.pth"},
    "B5_2X2":  {"weight_file": "weights/segformer_mit_b5_ade20k.pth"},
    "B5_HQ":   {"weight_file": "weights/segformer_mit_b5_ade20k.pth"},
}

# Official ADE20K URLs (mmseg) per variant
PRETRAIN_URLS: Dict[str, str] = {
    "B0": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_8x1_1024x1024_160k_ade20k/segformer_mit-b0_8x1_1024x1024_160k_ade20k_20210621_124307-57163b3b.pth",
    "B1": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_ade20k/segformer_mit-b1_8x1_1024x1024_160k_ade20k_20210621_124314-44ab0c0c.pth",
    "B2": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_8x1_1024x1024_160k_ade20k/segformer_mit-b2_8x1_1024x1024_160k_ade20k_20210621_124413-78e141dd.pth",
    "B3": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b3_8x1_1024x1024_160k_ade20k/segformer_mit-b3_8x1_1024x1024_160k_ade20k_20210621_124427-94dbd1b2.pth",
    "B4": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_ade20k/segformer_mit-b4_8x1_1024x1024_160k_ade20k_20210621_124600-5d0ce2a3.pth",
    "B5": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_ade20k/segformer_mit-b5_8x1_1024x1024_160k_ade20k_20210621_124600-ec54d796.pth",
    # aliases reuse B5 URL
    "B5_FAST": "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_ade20k/segformer_mit-b5_8x1_1024x1024_160k_ade20k_20210621_124600-ec54d796.pth",
    "B5_2X2":  "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_ade20k/segformer_mit-b5_8x1_1024x1024_160k_ade20k_20210621_124600-ec54d796.pth",
    "B5_HQ":   "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_8x1_1024x1024_160k_ade20k/segformer_mit-b5_8x1_1024x1024_160k_ade20k_20210621_124600-ec54d796.pth",
}


def _sanitize_pretrain_urls(urls: Dict[str, str]) -> Dict[str, str]:
    fixed = {}
    for k, v in urls.items():
        v = v.replace("/ssegformer/", "/segformer/")
        v = v.replace("/ssegformer_mit-", "/segformer_mit-")
        fixed[k] = v
    return fixed


# sanitise on import
PRETRAIN_URLS = _sanitize_pretrain_urls(PRETRAIN_URLS)


def variant_to_backbone_name(variant: str) -> str:
    v = variant.upper()
    if v.startswith("B5"):
        return "b5"  # B5 / B5_FAST / B5_2X2 / B5_HQ
    if v in ["B0", "B1", "B2", "B3", "B4"]:
        return v.lower()
    raise ValueError(f"Unknown variant: {variant}")


def filter_backbone_from_mmseg_ckpt(sd, msd):
    """
    Keep only backbone-compatible keys; skip decode/aux heads from mmseg ckpt.
    sd  = source state_dict (mmseg)
    msd = model.state_dict()
    """
    filt = {}
    for k, v in sd.items():
        if k.startswith("decode_head.") or k.startswith("aux_head."):
            continue
        mk = k if (k in msd) else (f"backbone.{k}" if f"backbone.{k}" in msd else None)
        if mk is not None and mk in msd and msd[mk].shape == v.shape:
            filt[mk] = v
    return filt
