"""
═══════════════════════════════════════════════════════════════════════════════════════
                            SEGFORMER ARCHITECTURE DIAGRAM
═══════════════════════════════════════════════════════════════════════════════════════

Input Image: RGB (B, 3, H, W) → Cityscapes: (B, 3, 1024, 2048)
                    ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    MiT (Mix Vision Transformer) Backbone                            │
│                                                                                     │
│  Stage 1: Patch Embed + Transformer Blocks                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Input: (B, 3, H, W) → (B, 3, 1024, 2048)                                  │   │
│  │ Patch Embed: 7×7 conv, stride=4, padding=3                                │   │
│  │ Output: (B, C1, H/4, W/4) → (B, 32/64, 256, 512)  [B0/B1-B5]            │   │
│  │                                                                            │   │
│  │ Multi-Head Attention Blocks × N1:                                         │   │
│  │ ├─ Efficient Attention (SRA): SR_ratio=8                                  │   │
│  │ ├─ Feed Forward Network                                                    │   │
│  │ └─ Layer Normalization + Residuals                                        │   │
│  │ Final: (B, C1, H/4, W/4) → (B, 32/64, 256, 512)                         │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                    ↓                                                               │
│  Stage 2: Downsample + Transformer Blocks                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Patch Merging: 3×3 conv, stride=2, padding=1                              │   │
│  │ Input: (B, C1, H/4, W/4) → Output: (B, C2, H/8, W/8)                     │   │
│  │ Tensor: (B, 64/128, 128, 256)  [B0/B1-B5]                                │   │
│  │                                                                            │   │
│  │ Multi-Head Attention Blocks × N2:                                         │   │
│  │ ├─ Efficient Attention (SRA): SR_ratio=4                                  │   │
│  │ ├─ Feed Forward Network                                                    │   │
│  │ └─ Layer Normalization + Residuals                                        │   │
│  │ Final: (B, C2, H/8, W/8) → (B, 64/128, 128, 256)                        │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                    ↓                                                               │
│  Stage 3: Downsample + Transformer Blocks                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Patch Merging: 3×3 conv, stride=2, padding=1                              │   │
│  │ Input: (B, C2, H/8, W/8) → Output: (B, C3, H/16, W/16)                   │   │
│  │ Tensor: (B, 160/320, 64, 128)  [B0/B1-B5]                                │   │
│  │                                                                            │   │
│  │ Multi-Head Attention Blocks × N3:                                         │   │
│  │ ├─ Efficient Attention (SRA): SR_ratio=2                                  │   │
│  │ ├─ Feed Forward Network                                                    │   │
│  │ └─ Layer Normalization + Residuals                                        │   │
│  │ Final: (B, C3, H/16, W/16) → (B, 160/320, 64, 128)                      │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                    ↓                                                               │
│  Stage 4: Downsample + Transformer Blocks                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Patch Merging: 3×3 conv, stride=2, padding=1                              │   │
│  │ Input: (B, C3, H/16, W/16) → Output: (B, C4, H/32, W/32)                 │   │
│  │ Tensor: (B, 256/512, 32, 64)  [B0/B1-B5]                                 │   │
│  │                                                                            │   │
│  │ Multi-Head Attention Blocks × N4:                                         │   │
│  │ ├─ Efficient Attention (SRA): SR_ratio=1 (No reduction)                   │   │
│  │ ├─ Feed Forward Network                                                    │   │
│  │ └─ Layer Normalization + Residuals                                        │   │
│  │ Final: (B, C4, H/32, W/32) → (B, 256/512, 32, 64)                       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                    ↓ Multi-scale Features
        F1: (B, C1, H/4, W/4)   F2: (B, C2, H/8, W/8)
        F3: (B, C3, H/16, W/16) F4: (B, C4, H/32, W/32)
                    ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Lightweight All-MLP Decoder                                │
│                                                                                     │
│  Step 1: Feature Projection                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Linear Projection (1×1 Conv): Each Fi → (B, 256, Hi, Wi)                  │   │
│  │ F1: (B, C1, H/4, W/4)   → (B, 256, H/4, W/4)   → (B, 256, 256, 512)     │   │
│  │ F2: (B, C2, H/8, W/8)   → (B, 256, H/8, W/8)   → (B, 256, 128, 256)     │   │
│  │ F3: (B, C3, H/16, W/16) → (B, 256, H/16, W/16) → (B, 256, 64, 128)      │   │
│  │ F4: (B, C4, H/32, W/32) → (B, 256, H/32, W/32) → (B, 256, 32, 64)       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                    ↓                                                               │
│  Step 2: Feature Upsampling & Alignment                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Bilinear Interpolation: Resize all to F1 resolution (H/4, W/4)            │   │
│  │ F1': (B, 256, H/4, W/4)   → (B, 256, 256, 512)  [No change]              │   │
│  │ F2': (B, 256, H/8, W/8)   → (B, 256, 256, 512)  [4× upsampling]          │   │
│  │ F3': (B, 256, H/16, W/16) → (B, 256, 256, 512)  [4× upsampling]          │   │
│  │ F4': (B, 256, H/32, W/32) → (B, 256, 256, 512)  [8× upsampling]          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                    ↓                                                               │
│  Step 3: Feature Fusion                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Concatenation: [F1', F2', F3', F4'] along channel dimension                │   │
│  │ Fused: (B, 1024, H/4, W/4) → (B, 1024, 256, 512)                         │   │
│  │                                                                            │   │
│  │ MLP Head: 3×3 Conv + BatchNorm + ReLU + Dropout                           │   │
│  │ Output: (B, 256, H/4, W/4) → (B, 256, 256, 512)                          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                    ↓                                                               │
│  Step 4: Final Classification                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Classification Head: 1×1 Conv                                              │   │
│  │ Input: (B, 256, H/4, W/4) → (B, 256, 256, 512)                           │   │
│  │ Output: (B, num_classes, H/4, W/4) → (B, 19, 256, 512)                   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                    ↓
            Final Upsampling (Training/Inference)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Bilinear Interpolation: (B, 19, H/4, W/4) → (B, 19, H, W)                         │
│ Final Output: (B, 19, 1024, 2048) for Cityscapes                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                              VARIANT-SPECIFIC CONFIGURATIONS
═══════════════════════════════════════════════════════════════════════════════════════

┌─────────┬────────┬────────┬─────────┬─────────┬─────────┬─────────┬─────────────────┐
│ Variant │ Depths │ Dims   │ Heads   │ SR_Ratio│ Params  │ FLOPs   │ Use Case        │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B0      │[2,2,2,2│[32,64, │[1,2,5,8]│[8,4,2,1]│ 3.8M    │ 8.4G    │ Mobile/Edge     │
│         │]       │160,256]│         │         │         │         │                 │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B1      │[2,2,2,2│[64,128,│[1,2,5,8]│[8,4,2,1]│ 14M     │ 16G     │ Efficient       │
│         │]       │320,512]│         │         │         │         │                 │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B2      │[3,4,6,3│[64,128,│[1,2,5,8]│[8,4,2,1]│ 25M     │ 62G     │ Balanced        │
│         │]       │320,512]│         │         │         │         │                 │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B3      │[3,4,18,│[64,128,│[1,2,5,8]│[8,4,2,1]│ 45M     │ 79G     │ High Quality    │
│         │3]      │320,512]│         │         │         │         │                 │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B4      │[3,8,27,│[64,128,│[1,2,5,8]│[8,4,2,1]│ 62M     │ 96G     │ Best Accuracy   │
│         │3]      │320,512]│         │         │         │         │                 │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B5      │[3,6,40,│[64,128,│[1,2,5,8]│[8,4,2,1]│ 82M     │ 101G    │ SOTA Performance│
│         │3]      │320,512]│         │         │         │         │                 │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B5_FAST │[3,6,40,│[64,128,│[1,2,5,8]│[8,4,2,1]│ 82M     │ 101G    │ Real-time Focus │
│         │3]      │320,512]│         │         │         │         │ + Token Merging │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B5_HQ   │[3,6,40,│[64,128,│[1,2,5,8]│[8,4,2,1]│ 82M     │ 101G    │ Quality Focus   │
│         │3]      │320,512]│         │         │         │         │ + Careful Merge │
├─────────┼────────┼────────┼─────────┼─────────┼─────────┼─────────┼─────────────────┤
│ B5_2X2  │[3,6,40,│[64,128,│[1,2,5,8]│[8,4,2,1]│ 82M     │ 101G    │ Balanced Merge  │
│         │3]      │320,512]│         │         │         │         │ + Structured 2×2│
└─────────┴────────┴────────┴─────────┴─────────┴─────────┴─────────┴─────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                          DETAILED ATTENTION MECHANISM (SRA)
═══════════════════════════════════════════════════════════════════════════════════════

For Stage i with SR_ratio = Ri:

Input: X ∈ ℝ^(B × N × C), where N = (H/2^(i+1)) × (W/2^(i+1))

1. Query Computation:
   Q = X × W_Q ∈ ℝ^(B × N × C)

2. Spatial Reduction (Key & Value):
   - Reshape: X → ℝ^(B × H' × W' × C), where H'×W' = N
   - SR Conv: Conv2d(C, C, kernel=Ri, stride=Ri) + LayerNorm
   - X_reduced ∈ ℝ^(B × (H'/Ri) × (W'/Ri) × C)
   - Flatten: X_reduced → ℝ^(B × (N/Ri²) × C)
   
3. Key & Value Computation:
   K = X_reduced × W_K ∈ ℝ^(B × (N/Ri²) × C)
   V = X_reduced × W_V ∈ ℝ^(B × (N/Ri²) × C)

4. Multi-Head Attention:
   Attention(Q,K,V) = Softmax(QK^T/√(C/h)) V
   where h = number of heads

Memory Reduction: N × N → N × (N/Ri²)
For Cityscapes B5 Stage 1: 131072 × 131072 → 131072 × 2048 (64× reduction!)

═══════════════════════════════════════════════════════════════════════════════════════
                    FUN-LORA ALGORITHM: DYNAMIC INFORMATION INJECTION
═══════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: INITIALIZATION                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

Step 1: LoRA Injection on Frozen Backbone
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Original Linear Layer: W ∈ ℝ^(d_out × d_in)                                       │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ Input x ∈ ℝ^(B × L × d_in) → W·x → Output ∈ ℝ^(B × L × d_out)            │   │
│ │ Status: W.requires_grad = False (FROZEN)                                   │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓ LoRA Injection                                │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ Enhanced Layer: W + ΔW where ΔW = B·A                                     │   │
│ │                                                                            │   │
│ │ LoRA Branch:                                                               │   │
│ │ Input x → A ∈ ℝ^(d_in × r) → B ∈ ℝ^(r × d_out) → α/r · ΔW·x             │   │
│ │                                                                            │   │
│ │ Combined Output: W·x + (α/r)·B·A·x                                        │   │
│ │ Status: W.requires_grad = False, A.requires_grad = True, B.requires_grad = True│   │
│ │ Parameters: A ~ Kaiming, B ~ Zero (initially no change to forward pass)   │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘

Target Layers for LoRA Injection:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: qkv, proj, fc1, fc2 → 4 LoRA adapters per block × 2 blocks = 8 adapters  │
│ Stage 2: qkv, proj, fc1, fc2 → 4 LoRA adapters per block × 2 blocks = 8 adapters  │
│ Stage 3: qkv, proj, fc1, fc2 → 4 LoRA adapters per block × 18 blocks = 72 adapters│
│ Stage 4: qkv, proj, fc1, fc2 → 4 LoRA adapters per block × 3 blocks = 12 adapters │
│ Total: ~100 LoRA adapters injected across the backbone                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: TEMPORAL TRAINING EVOLUTION                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

Epochs 1-10: Head Warmup Phase
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Trainable Parameters:                                                               │
│ ✓ Decoder Head (proj, fuse, classifier): ~1M params @ lr=4e-5                     │
│ ✓ LoRA Adapters (A, B matrices): ~200K params @ lr=5e-5                           │
│ ✗ Backbone Weights (W matrices): ~80M params @ lr=0.0 (FROZEN)                    │
│                                                                                     │
│ Information Flow:                                                                   │
│ Input → [Frozen Backbone + Active LoRA] → [Trainable Decoder] → Output            │
│                                                                                     │
│ Gradient Flow:                                                                      │
│ Loss ← Decoder ← Multi-scale Features ← [∂LoRA/∂A, ∂LoRA/∂B] ← Frozen Backbone    │
│                                                                                     │
│ Learning Dynamics:                                                                  │
│ • LoRA adapters learn task-specific low-rank residuals: ΔW = B·A                  │
│ • Decoder learns to fuse multi-scale features for Cityscapes classes              │
│ • Backbone features remain at ADE20K initialization (knowledge preservation)       │
└─────────────────────────────────────────────────────────────────────────────────────┘

Epoch 10: Classic Backbone Unfreezing
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Trainable Parameters:                                                               │
│ ✓ Decoder Head: ~1M params @ lr=4e-5                                              │
│ ✓ LoRA Adapters: ~200K params @ lr=5e-5                                           │
│ ✓ Backbone Weights: ~80M params @ lr=1e-5 (UNFROZEN)                             │
│                                                                                     │
│ Information Flow Change:                                                            │
│ Input → [Active Backbone + Active LoRA] → [Active Decoder] → Output               │
│                                                                                     │
│ Gradient Flow Enhancement:                                                          │
│ Loss ← Decoder ← Multi-scale Features ← [∂W/∂θ + ∂LoRA/∂A + ∂LoRA/∂B] ← Input    │
│                                                                                     │
│ Learning Rate Hierarchy:                                                            │
│ lr_head > lr_lora > lr_backbone (4e-5 > 5e-5 > 1e-5)                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

Epochs 13, 16, 19, ...: Fisher-Guided Unfreezing (Every 3 epochs)
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Fisher Information Computation                                             │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ Temporarily enable gradients on ALL frozen backbone groups                 │   │
│ │ Groups identified: {stage_i.block_j, layer_k, encoder_m} modules          │   │
│ │                                                                            │   │
│ │ For b = 1 to fisher_batches (12 mini-batches):                           │   │
│ │   Sample (x, y) from training data                                        │   │
│ │   Compute: ℒ = CrossEntropyLoss(model(x), y)                             │   │
│ │   Backpropagate: ∇_θ ℒ                                                   │   │
│ │   Accumulate Fisher scores: S_m += ||∇_θ_m ℒ||²_2                       │   │
│ │   Zero gradients: ∇_θ ← 0                                                │   │
│ │                                                                            │   │
│ │ Fisher Scores Example:                                                     │   │
│ │ stage_3.block_15: S = 2.341e-4 (High information)                        │   │
│ │ stage_2.block_3:  S = 1.892e-4 (Medium information)                      │   │
│ │ stage_1.block_1:  S = 4.521e-5 (Low information)                         │   │
│ │ stage_4.block_2:  S = 3.112e-5 (Low information)                         │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│ STEP 2: Information-Guided Selection                                               │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ Rank groups by Fisher score (descending)                                   │   │
│ │ Select top max_groups_per_unfreeze (1) groups for unfreezing              │   │
│ │ Chosen: stage_3.block_15 (highest Fisher score)                           │   │
│ │                                                                            │   │
│ │ Unfreeze selected group:                                                   │   │
│ │ stage_3.block_15.{qkv,proj,fc1,fc2}.weight.requires_grad = True          │   │
│ │                                                                            │   │
│ │ Keep others frozen:                                                        │   │
│ │ stage_{1,2,4}.*.weight.requires_grad = False                              │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│ STEP 3: Dynamic Parameter Group Reconstruction                                     │
│ ┌─────────────────────────────────────────────────────────────────────────────┐   │
│ │ Rebuild optimizer with updated parameter groups:                           │   │
│ │                                                                            │   │
│ │ Group 1 - Head: {decoder.proj, decoder.fuse, decoder.classifier}         │   │
│ │          Parameters: ~1M, Learning Rate: 4e-5                             │   │
│ │                                                                            │   │
│ │ Group 2 - LoRA: {*.lora_A, *.lora_B} (all stages)                        │   │
│ │          Parameters: ~200K, Learning Rate: 5e-5                           │   │
│ │                                                                            │   │
│ │ Group 3 - Backbone: {stage_3.block_15.*} (newly unfrozen)                │   │
│ │          Parameters: ~1.5M, Learning Rate: 1e-5                           │   │
│ │                                                                            │   │
│ │ Total Trainable: ~2.7M / ~82M (3.3% of total parameters)                 │   │
│ └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE INFORMATION INJECTION TIMELINE                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

Epoch 1-10: Foundation Learning
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────┐
│ Stage 1         │ Stage 2         │ Stage 3         │ Stage 4         │ Decoder     │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┤
│ W: FROZEN       │ W: FROZEN       │ W: FROZEN       │ W: FROZEN       │ W: ACTIVE   │
│ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: N/A   │
│ Info: ADE20K    │ Info: ADE20K    │ Info: ADE20K    │ Info: ADE20K    │ Info: Learn │
│ Flow: ────→     │ Flow: ────→     │ Flow: ────→     │ Flow: ────→     │ Flow: ←──── │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────┘

Epoch 10: Classic Unfreezing
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────┐
│ Stage 1         │ Stage 2         │ Stage 3         │ Stage 4         │ Decoder     │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┤
│ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE   │
│ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: N/A   │
│ Info: ADE→CS    │ Info: ADE→CS    │ Info: ADE→CS    │ Info: ADE→CS    │ Info: Learn │
│ Flow: ←──→      │ Flow: ←──→      │ Flow: ←──→      │ Flow: ←──→      │ Flow: ←──── │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────┘

Epoch 13: First Fisher-Guided Unfreezing
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────┐
│ Stage 1         │ Stage 2         │ Stage 3         │ Stage 4         │ Decoder     │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┤
│ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE   │
│ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: N/A   │
│ Info: ADE→CS    │ Info: ADE→CS    │ Info: ADE→CS↑   │ Info: ADE→CS    │ Info: Learn │
│ Flow: ←──→      │ Flow: ←──→      │ Flow: ←══→      │ Flow: ←──→      │ Flow: ←──── │
│ Fisher: 0.045   │ Fisher: 0.189   │ Fisher: 0.234★  │ Fisher: 0.031   │ N/A         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────┘
★ = Selected for enhanced learning (highest Fisher score)

Epoch 16: Second Fisher-Guided Unfreezing
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────┐
│ Stage 1         │ Stage 2         │ Stage 3         │ Stage 4         │ Decoder     │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┤
│ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE       │ W: ACTIVE   │
│ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: ACTIVE    │ LoRA: N/A   │
│ Info: ADE→CS    │ Info: ADE→CS↑   │ Info: ADE→CS↑↑  │ Info: ADE→CS    │ Info: Learn │
│ Flow: ←──→      │ Flow: ←══→      │ Flow: ←═══→     │ Flow: ←──→      │ Flow: ←──── │
│ Fisher: 0.067   │ Fisher: 0.201★  │ Fisher: 0.198   │ Fisher: 0.029   │ N/A         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┴─────────────┘
★ = Selected for enhanced learning, ↑ = Previous enhancement level

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        INFORMATION ADAPTATION MECHANISMS                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

1. LoRA Residual Learning:
   Original: y = W·x
   Enhanced: y = W·x + (α/r)·B·A·x
   
   Where: ΔW = B·A represents low-rank task-specific adaptations
   Effect: Preserves pre-trained knowledge while adding domain-specific modifications

2. Fisher-Guided Attention:
   F_i = ||∇_θ_i ℒ||²_2 measures information importance per module
   High F_i → High information content → Priority for unfreezing
   Effect: Focuses computational resources on information-rich regions

3. Hierarchical Learning Rates:
   lr_head >> lr_lora > lr_backbone
   Effect: Rapid task adaptation → Fine-grained adaptation → Careful knowledge transfer

4. Progressive Complexity:
   Epochs 1-10:   Simple head adaptation (1.2M params)
   Epochs 10+:    Full model adaptation (82M params)
   Epochs 13+:    Information-guided selective enhancement
   Effect: Stable learning progression from simple to complex

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          MODEL STATE EVOLUTION SUMMARY                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

Parameter Evolution:
┌──────────┬──────────────┬──────────────┬──────────────┬─────────────────────────┐
│ Epoch    │ Trainable    │ LoRA Active  │ Backbone     │ Information Content     │
├──────────┼──────────────┼──────────────┼──────────────┼─────────────────────────┤
│ 1-10     │ 1.2M (1.5%)  │ 200K (0.2%)  │ Frozen       │ Task adaptation         │
│ 10       │ 82M (100%)   │ 200K (0.2%)  │ Active       │ Full knowledge transfer │
│ 13       │ 82M + Focus  │ 200K (0.2%)  │ Selective    │ Information-guided      │
│ 16       │ 82M + Focus+ │ 200K (0.2%)  │ Selective++  │ Enhanced information    │
│ ...      │ Progressive  │ Stable       │ Dynamic      │ Iterative refinement    │
└──────────┴──────────────┴──────────────┴──────────────┴─────────────────────────┘

Information Flow Evolution:
Phase 1: ADE20K Features → [LoRA] → Cityscapes Head
Phase 2: ADE20K→CS Features → [LoRA] → Enhanced Decoder  
Phase 3: ADE20K→CS Features → [Fisher-guided LoRA] → Optimized Output

Final State: Cityscapes-optimized SegFormer with preserved ADE20K knowledge base

═══════════════════════════════════════════════════════════════════════════════════════
"""
