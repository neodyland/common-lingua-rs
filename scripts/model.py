import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ByteNgramEmbed(nn.Module):
    def __init__(self, num_buckets=4096, embed_dim=64, n=3):
        super().__init__()
        self.n = n
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        clamped = byte_ids.clamp(max=255)
        padded = F.pad(clamped, (0, self.n - 1), value=0)
        h = torch.zeros(B, T, dtype=torch.long, device=byte_ids.device)
        for i in range(self.n):
            h = h * 257 + padded[:, i : i + T]
        return self.embed(h % self.num_buckets)


class ByteConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=15, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x).transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = F.silu(self.conv(x)).transpose(1, 2)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return residual + x


def _rope(q, k):
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    freqs = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=q.device)
    a = torch.outer(t, freqs)
    cos = a.cos().to(q.dtype)
    sin = a.sin().to(q.dtype)

    def rot(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


class ByteAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, expand=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        residual = x
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        q, k = _rope(q, k)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.out_proj(out)

        residual = x
        h = self.norm2(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return residual + h


class ByteHybrid(nn.Module):
    def __init__(
        self,
        num_classes,
        d_model=256,
        n_conv=3,
        n_attn=1,
        n_heads=4,
        ffn_expand=2,
        max_len=512,
        conv_kernel=15,
        ngram_buckets=0,
        ngram_dim=64,
    ):
        super().__init__()
        self.max_len = max_len

        self.embed = nn.Embedding(257, d_model, padding_idx=256)

        self.ngram_embed = None
        if ngram_buckets > 0:
            self.ngram_embed = ByteNgramEmbed(ngram_buckets, ngram_dim, n=3)
            self.ngram_proj = nn.Linear(ngram_dim, d_model, bias=False)

        self.conv_layers = nn.ModuleList(
            [ByteConvBlock(d_model, conv_kernel, ffn_expand) for _ in range(n_conv)]
        )
        self.attn_layers = nn.ModuleList(
            [ByteAttnBlock(d_model, n_heads, ffn_expand) for _ in range(n_attn)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, byte_ids):
        pad_mask = byte_ids != 256
        x = self.embed(byte_ids)
        if self.ngram_embed is not None:
            x = x + self.ngram_proj(self.ngram_embed(byte_ids))
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.final_norm(x)
        mask = pad_mask.unsqueeze(-1).to(x.dtype)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(x)


def load(checkpoint, dtype="fp32", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ByteHybrid(
        num_classes=ckpt["num_classes"],
        max_len=ckpt["max_len"],
        d_model=256, n_conv=3, n_attn=1, n_heads=4, conv_kernel=15, ngram_buckets=4096, ngram_dim=64,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval().to(device)
    idx2lang = {v: k for k, v in ckpt["lang2idx"].items()}
    return model, idx2lang, ckpt["max_len"], device


def encode(texts, max_len):
    out = np.full((len(texts), max_len), 256, dtype=np.int64)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        raw = t.encode("utf-8", errors="replace")[:max_len]
        if raw:
            out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return torch.from_numpy(out)


@torch.no_grad()
def predict(model, texts, idx2lang, max_len, device, top_k=3, batch_size=256):
    """Returns a list of [(lang, prob), ...] (one list per text, top-k entries each)."""
    out = []
    for i in range(0, len(texts), batch_size):
        b = encode(texts[i : i + batch_size], max_len).to(device)
        probs = torch.softmax(model(b).float(), dim=-1)
        top_p, top_idx = probs.topk(top_k, dim=-1)
        for p_row, idx_row in zip(top_p.cpu().tolist(), top_idx.cpu().tolist()):
            out.append([(idx2lang[j], float(p)) for p, j in zip(p_row, idx_row)])
    return out


def _main():
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("text", nargs="*", help="Text(s) to classify (one per arg).")
    args = p.parse_args()
    texts = args.text
    device = "cpu"
    dtype = "fp32"
    model, idx2lang, max_len, device = load("./model.pt", dtype, device)
    print(
        f"# {len(idx2lang)} languages, max_len={max_len}, dtype={dtype}, device={device}",
    )
    for text, top in zip(
        texts,
        predict(model, texts, idx2lang, max_len, device, 3, 256),
    ):
        preview = text[:80].replace("\n", " ")
        others = " ".join(f"{lg}={p:.3f}" for lg, p in top[1:])
        print(f"{top[0][0]}\t{top[0][1]:.4f}\t{others}\t{preview}")


if __name__ == "__main__":
    _main()
