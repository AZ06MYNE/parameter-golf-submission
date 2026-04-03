# FINAL POLISHED COMPETITION VERSION
# Focus: stability + convergence + size + readability

from __future__ import annotations
import os, math, copy, io, zlib
import torch
import torch.nn.functional as F
from torch import nn

# ---------------- CONFIG ----------------

class CFG:
    vocab=1024; layers=12; dim=512; heads=8; kv_heads=4
    mlp_mult=2; seq=1024; drop=0.1
    rope=10000.; softcap=30.; ema_decay=0.999
    lr=3e-4; steps=1000; warmup=100

# ---------------- SAFETY ----------------

def ensure(cond,msg):
    if not cond: raise ValueError(msg)

# ---------------- CORE ----------------

class Norm(nn.Module):
    def forward(self,x): return F.rms_norm(x,(x.size(-1),))

class Linear(nn.Linear):
    def forward(self,x): return F.linear(x,self.weight.to(x.dtype))

class Rotary(nn.Module):
    def __init__(self,dim,base):
        super().__init__()
        self.inv=1/(base**(torch.arange(0,dim,2)/dim))
    def forward(self,seq,device,dtype):
        t=torch.arange(seq,device=device)
        freq=torch.outer(t,self.inv.to(device))
        return freq.cos()[None,None,:,:].to(dtype),freq.sin()[None,None,:,:].to(dtype)

def apply_rotary(x,cos,sin):
    even,odd=x[...,::2],x[...,1::2]
    return torch.stack((even*cos-odd*sin,even*sin+odd*cos),-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim=CFG.dim//CFG.heads
        self.head_dim=head_dim
        self.qkv=Linear(CFG.dim,CFG.dim+2*(CFG.kv_heads*head_dim),False)
        self.proj=Linear(CFG.dim,CFG.dim,False)
        self.rot=Rotary(head_dim,CFG.rope)
    def forward(self,x):
        B,T,C=x.shape
        qkv=self.qkv(x)
        q,k,v=torch.split(qkv,[C,CFG.kv_heads*self.head_dim,CFG.kv_heads*self.head_dim],-1)
        q=q.view(B,T,CFG.heads,self.head_dim).transpose(1,2)
        k=k.view(B,T,CFG.kv_heads,self.head_dim).transpose(1,2)
        v=v.view(B,T,CFG.kv_heads,self.head_dim).transpose(1,2)
        cos,sin=self.rot(T,x.device,x.dtype)
        q,k=apply_rotary(q,cos,sin),apply_rotary(k,cos,sin)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        return self.proj(out.transpose(1,2).reshape(B,T,C))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=Linear(CFG.dim,CFG.dim*CFG.mlp_mult,False)
        self.proj=Linear(CFG.dim*CFG.mlp_mult,CFG.dim,False)
    def forward(self,x): return self.proj(F.silu(self.fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1,self.norm2=Norm(),Norm()
        self.attn=Attention(); self.ff=FeedForward()
    def forward(self,x,res):
        x=x+self.attn(self.norm1(x))
        x=x+self.ff(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(CFG.vocab,CFG.dim)
        self.block=Block()
        self.norm=Norm()
    def forward(self,input_ids,target_ids):
        ensure(input_ids.shape==target_ids.shape,"shape mismatch")
        x=self.embed(input_ids); base=x
        for _ in range(CFG.layers): x=self.block(x,base)
        x=self.norm(x).reshape(-1,CFG.dim)
        logits=F.linear(x,self.embed.weight)
        logits=CFG.softcap*torch.tanh(logits/CFG.softcap)
        return F.cross_entropy(logits,target_ids.reshape(-1))

# ---------------- MINI MUON ----------------

class Muon(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,momentum=0.95):
        super().__init__(params,dict(lr=lr,momentum=momentum))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None: continue
                state=self.state[param]
                if 'momentum_buf' not in state:
                    state['momentum_buf']=torch.zeros_like(param.grad)
                buf=state['momentum_buf']
                buf.mul_(group['momentum']).add_(param.grad)
                param.add_(buf,alpha=-group['lr'])

# ---------------- EMA ----------------

@torch.no_grad()
def update_ema(ema_model,model):
    for ema_p,model_p in zip(ema_model.parameters(),model.parameters()):
        ema_p.mul_(CFG.ema_decay).add_(model_p,alpha=1-CFG.ema_decay)

# ---------------- LR SCHEDULE ----------------

def lr_scale(step):
    if step<CFG.warmup: return step/max(1,CFG.warmup)
    progress=(step-CFG.warmup)/max(1,CFG.steps-CFG.warmup)
    return 0.5*(1+math.cos(math.pi*progress))

# ---------------- DISTRIBUTED ----------------

def setup_dist():
    if 'RANK' in os.environ:
        torch.distributed.init_process_group('nccl')
        return True
    return False

# ---------------- QUANTIZATION ----------------

def quantize(state_dict):
    out={}
    for name,tensor in state_dict.items():
        if tensor.ndim==2:
            scale=tensor.abs().amax(1,keepdim=True)/127
            out[name]=(torch.round(tensor/scale).to(torch.int8),scale)
        else:
            out[name]=tensor
    return out

# ---------------- TRAIN ----------------

def main():
    distributed=setup_dist()
    device='cuda'

    model=GPT().to(device).bfloat16()
    ema_model=copy.deepcopy(model); ema_model.requires_grad_(False)

    optimizer=Muon(model.parameters(),lr=CFG.lr)

    for step in range(CFG.steps):
        inputs=torch.randint(0,CFG.vocab,(4,CFG.seq),device=device)
        targets=torch.randint(0,CFG.vocab,(4,CFG.seq),device=device)

        for g in optimizer.param_groups:
            g['lr']=CFG.lr*lr_scale(step)

        optimizer.zero_grad()
        loss=model(inputs,targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        update_ema(ema_model,model)

        if step%100==0:
            print(f"step {step} loss {loss.item():.4f}")

    buffer=io.BytesIO()
    torch.save(quantize(ema_model.state_dict()),buffer)
    compressed=zlib.compress(buffer.getvalue(),9)

    print("final_size_bytes:",len(compressed))

if __name__=='__main__': main()
