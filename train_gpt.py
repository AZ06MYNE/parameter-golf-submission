"""
Merged Parameter Golf Architecture
Combines the extreme leanness of Code 1 with the stability, structure, and real Muon optimizer of Code 2.
"""

from __future__ import annotations
import os, math, copy, io, zlib, glob, time, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# CONFIGURATION
# -----------------------------

class CFG:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    
    vocab = 1024
    layers = 12
    dim = 512
    heads = 8
    kv_heads = 4
    mlp_mult = 2
    seq = 1024
    drop = 0.1
    rope = 10000.0
    softcap = 30.0
    ema_decay = 0.999
    
    lr_embed = 0.05
    lr_matrix = 0.04
    lr_scalar = 0.04
    warmup_steps = 250
    iterations = 20000
    train_batch_tokens = 524_288
    val_batch_size = 524_288

# -----------------------------
# CORE MODULES & MATH (From Code 1)
# -----------------------------

class Norm(nn.Module):
    def forward(self, x: Tensor) -> Tensor: 
        return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.inv = 1 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        
    def forward(self, seq: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq, device=device, dtype=torch.float32)
        freq = torch.outer(t, self.inv.to(device))
        return freq.cos()[None, None, :, :].to(dtype), freq.sin()[None, None, :, :].to(dtype)

def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Code 1's elegant interleaved rotary application
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

# -----------------------------
# TRANSFORMER ARCHITECTURE
# -----------------------------

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = CFG.dim // CFG.heads
        kv_dim = CFG.kv_heads * self.head_dim
        
        self.qkv = nn.Linear(CFG.dim, CFG.dim + 2 * kv_dim, bias=False)
        self.proj = nn.Linear(CFG.dim, CFG.dim, bias=False)
        self.proj.weight.data.zero_() # Zero init for stability
        
        self.q_gain = nn.Parameter(torch.full((CFG.heads,), 1.5, dtype=torch.float32))
        self.rot = Rotary(self.head_dim, CFG.rope)
        self.drop = nn.Dropout(CFG.drop)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        
        q_size = CFG.heads * self.head_dim
        kv_size = CFG.kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        
        q = q.view(B, T, CFG.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, CFG.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, CFG.kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rot(T, x.device, x.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            enable_gqa=(CFG.heads != CFG.kv_heads),
            dropout_p=CFG.drop if self.training else 0.0
        )
        return self.drop(self.proj(out.transpose(1, 2).reshape(B, T, C)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = CFG.dim * CFG.mlp_mult
        self.fc = nn.Linear(CFG.dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, CFG.dim, bias=False)
        self.proj.weight.data.zero_()
        self.drop = nn.Dropout(CFG.drop)

    def forward(self, x: Tensor) -> Tensor:
        # SiLU for smoother gradients
        return self.drop(self.proj(F.silu(self.fc(x))))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = Norm(), Norm()
        self.attn = Attention()
        self.ff = FeedForward()
        
        # Trainable scaling and mixing for stability in shared-weight loops
        self.attn_scale = nn.Parameter(torch.ones(CFG.dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(CFG.dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(CFG.dim), torch.zeros(CFG.dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        
        attn_out = self.attn(self.norm1(x))
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.ff(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Token and Symmetry-breaking Layer Embeddings
        self.embed = nn.Embedding(CFG.vocab, CFG.dim)
        self.layer_emb = nn.Embedding(CFG.layers, CFG.dim)
        nn.init.normal_(self.embed.weight, std=0.005)
        nn.init.normal_(self.layer_emb.weight, std=0.005)
        
        self.block = Block() # Shared block for depth recurrence
        self.norm = Norm()
        self.drop = nn.Dropout(CFG.drop)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.drop(F.rms_norm(self.embed(input_ids), (CFG.dim,)))
        x0 = x 
        
        # Recurrent Loop over the same block (Parameter Golf)
        for r in range(CFG.layers):
            layer_id = torch.tensor(r, device=x.device)
            x_in = x + self.layer_emb(layer_id).to(x.dtype)
            x = self.block(x_in, x0)

        x = self.norm(x).reshape(-1, CFG.dim)
        
        # Tied embeddings
        logits = F.linear(x, self.embed.weight)
        logits = CFG.softcap * torch.tanh(logits / CFG.softcap)
        return F.cross_entropy(logits, target_ids.reshape(-1))

# -----------------------------
# OPTIMIZERS (Real Muon from Code 2)
# -----------------------------

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                
                # Nesterov + Newton-Schulz orthogonalization
                g = g.add(buf, alpha=group["momentum"])
                g = zeropower_via_newtonschulz5(g)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"])

# -----------------------------
# UTILS & DATA LOADING
# -----------------------------

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module):
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(CFG.ema_decay).add_(model_p.data, alpha=1 - CFG.ema_decay)

def lr_scale(step: int) -> float:
    if step < CFG.warmup_steps: return step / max(1, CFG.warmup_steps)
    progress = (step - CFG.warmup_steps) / max(1, CFG.iterations - CFG.warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress)) if progress < 1.0 else 0.0

def load_data_shard(file: Path) -> Tensor:
    # Fineweb fast binary loading
    header_bytes = 256 * 4
    header = np.fromfile(file, dtype="<i4", count=256)
    tokens_np = np.fromfile(file, dtype="<u2", count=int(header[2]), offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False)).to(torch.int64)

# -----------------------------
# LEAN QUANTIZATION (From Code 1)
# -----------------------------

def quantize_lean(state_dict: dict) -> dict:
    """Elegant, highly readable 8-bit quantization for submission sizing."""
    out = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float()
        if t.ndim == 2:
            scale = (t.abs().amax(1, keepdim=True) / 127).clamp_min(1e-5)
            out[name] = (torch.round(t / scale).to(torch.int8), scale.half())
        else:
            out[name] = t.half() # Keep 1D vectors in FP16
    return out

# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------

def main():
    distributed = "RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if distributed: dist.init_process_group("nccl")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = GPT().to(device).bfloat16()
    ema_model = copy.deepcopy(model).requires_grad_(False)
    
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Parameter grouping for optimizers
    matrix_params, scalar_params = [], []
    for name, p in model.named_parameters():
        if "embed" in name or "layer_emb" in name: continue
        if p.ndim == 2: matrix_params.append(p)
        else: scalar_params.append(p)

    opt_embed = torch.optim.Adam([model.module.embed.weight if distributed else model.embed.weight, 
                                  model.module.layer_emb.weight if distributed else model.layer_emb.weight], 
                                  lr=CFG.lr_embed, betas=(0.9, 0.95))
    opt_muon = Muon(matrix_params, lr=CFG.lr_matrix)
    opt_scalar = torch.optim.Adam(scalar_params, lr=CFG.lr_scalar, betas=(0.9, 0.95))
    optimizers = [opt_embed, opt_muon, opt_scalar]

    # Dummy Data setup (Replace with DistributedTokenLoader in production)
    # Using random data here to keep the script self-contained and runnable
    print("Beginning Training...")
    
    for step in range(CFG.iterations):
        # Generate dummy batch
        x = torch.randint(0, CFG.vocab, (4, CFG.seq), device=device)
        y = torch.randint(0, CFG.vocab, (4, CFG.seq), device=device)

        scale = lr_scale(step)
        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g.get("base_lr", g["lr"]) * scale

        for opt in optimizers: opt.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        for opt in optimizers: opt.step()
        
        update_ema(ema_model, model.module if distributed else model)

        if step % 100 == 0 and local_rank == 0:
            print(f"Step {step:05d} | Loss: {loss.item():.4f}")

    # Compress and Save Model
    if local_rank == 0:
        print("\nQuantizing and Compressing...")
        quantized_dict = quantize_lean(ema_model.state_dict())
        buffer = io.BytesIO()
        torch.save(quantized_dict, buffer)
        compressed_blob = zlib.compress(buffer.getvalue(), level=9)
        
        with open("submission.ptz", "wb") as f:
            f.write(compressed_blob)
            
        print(f"Final Output Size: {len(compressed_blob) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
