import torch
from safetensors import torch as st


def quant_row_lsq(src: torch.Tensor):
    mx = src.abs().max().item()
    if mx <= 0.0:
        return torch.zeros_like(src, dtype=torch.int8), 0.0

    inv = 127.0 / mx
    q = torch.round(src * inv).clamp_(-127, 127).to(torch.int8)

    qf = q.to(torch.float32)
    d1 = (src * qf).sum().item()
    d2 = (qf * qf).sum().item()

    scale = d1 / d2 if d2 > 0.0 else mx * (1.0 / 127.0)
    return q, scale


class QMat:
    def __init__(self, w: torch.Tensor, scale: torch.Tensor):
        self.w = w
        self.scale = scale

    @classmethod
    def quant(cls, w: torch.Tensor):
        mx = w.abs().amax(dim=1)
        inv = torch.where(mx > 0, 127.0 / mx, torch.zeros_like(mx))
        q = torch.round(w * inv.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
        qf = q.to(torch.float32)
        d1 = (w * qf).sum(dim=1)
        d2 = (qf * qf).sum(dim=1)
        scale = torch.where(d2 > 0, d1 / d2.clamp_min(1e-30), mx / 127.0)
        scale = torch.where(mx > 0, scale, torch.zeros_like(scale))
        return cls(q, scale)


def main():
    checkpoint = torch.load("./model.pt", map_location="cpu", weights_only=False)
    state = {}
    for name, tensor in checkpoint["model_state_dict"].items():
        if (
            "qkv" not in name
            and "out_proj" not in name
            and "ffn" not in name
            and "ngram_proj" not in name
        ):
            state[name] = tensor
            continue
        name = name[: -len("weight")]
        mat = QMat.quant(tensor)
        state[f"{name}w"] = mat.w
        state[f"{name}scale"] = mat.scale
    st.save_file(state, "../common-lingua/src/weight.safetensors")


if __name__ == "__main__":
    main()
