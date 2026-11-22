"""PyTorch PP-OCRv6 medium recognizer (CTC inference and training path).

Architecture follows PaddleOCR's rec_lcnetv4.py, rnn.py, rec_svtrnet.py,
rec_multi_head.py, and PP-OCRv6_medium_rec.yml.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


REC_CONFIG = {
    "tiny": {
        "stem": (24, 48), "stem_type": "simple",
        "blocks2": [(3, 48, 48, 1, True)],
        "blocks3": [(3, 48, 48, 1, False)],
        "blocks4": [(3, 48, 96, (2, 1), False), (3, 96, 96, 1, True), (3, 96, 96, 1, False)],
        "blocks5": [(3, 96, 160, (2, 1), False), (3, 160, 160, 1, True), (3, 160, 160, 1, False), (3, 160, 160, 1, False)],
        "blocks6": [],
    },
    "small": {
        "stem": (48, 96), "stem_type": "branch",
        "blocks2": [(3, 96, 96, 1, True)],
        "blocks3": [(3, 96, 96, 1, False), (3, 96, 96, 1, False)],
        "blocks4": [(3, 96, 192, (2, 1), False), (3, 192, 192, 1, True), (3, 192, 192, 1, False), (3, 192, 192, 1, True), (3, 192, 192, 1, False), (3, 192, 192, 1, True), (3, 192, 192, 1, False)],
        "blocks5": [(3, 192, 384, (2, 1), False), (3, 384, 384, 1, True), (3, 384, 384, 1, False)],
        "blocks6": [],
    },
    "medium": {
        "stem": (64, 128), "stem_type": "branch",
        "blocks2": [(3, 128, 128, 1, True)],
        "blocks3": [(3, 128, 256, 1, False), (3, 256, 256, 1, False), (3, 256, 256, 1, True)],
        "blocks4": [(3, 256, 512, (2, 1), False), (3, 512, 512, 1, True), (3, 512, 512, 1, False), (3, 512, 512, 1, True), (3, 512, 512, 1, False), (3, 512, 512, 1, True), (3, 512, 512, 1, False)],
        "blocks5": [(3, 512, 768, (2, 1), False), (3, 768, 768, 1, True), (3, 768, 768, 1, False)],
        "blocks6": [],
    },
}


def _stride_pair(stride: int | Sequence[int]) -> tuple[int, int]:
    return (stride, stride) if isinstance(stride, int) else tuple(stride)


def _kernel_pair(kernel: int | Sequence[int]) -> tuple[int, int]:
    return (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)


def _padding_pair(kernel: int | Sequence[int]) -> tuple[int, int]:
    kh, kw = _kernel_pair(kernel)
    return (kh // 2, kw // 2)


class ConvBN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        groups: int = 1,
        padding: str | tuple[int, int] = "explicit",
        use_bias: bool = True,
        bn_scale_init: float = 1.0,
    ):
        super().__init__()
        k_pair = _kernel_pair(kernel)
        s_pair = _stride_pair(stride)
        p_pair = _padding_pair(kernel) if padding == "explicit" else (0, 0)
        self.use_custom_even_padding = (padding == "same" and k_pair == (2, 2))

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            k_pair,
            stride=s_pair,
            padding=p_pair if not self.use_custom_even_padding else (0, 0),
            groups=groups,
            bias=use_bias,
        )
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        nn.init.constant_(self.bn.weight, bn_scale_init)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x: torch.Tensor, activation: str | None = None) -> torch.Tensor:
        if self.use_custom_even_padding:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        x = self.bn(x)
        if activation == "relu":
            return F.relu(x)
        if activation == "gelu":
            return F.gelu(x)
        if activation == "silu":
            return F.silu(x)
        if activation == "hardswish":
            return F.hardswish(x)
        return x


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        groups: int = 1,
        padding: str | tuple[int, int] = "explicit",
        use_act: bool = True,
    ):
        super().__init__()
        self.use_act = use_act
        self.block = ConvBN(
            in_ch, out_ch, kernel, stride=stride, groups=groups, padding=padding, use_bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x, activation="relu" if self.use_act else None)


class StemBlock(nn.Module):
    def __init__(self, mid_ch: int, out_ch: int):
        super().__init__()
        self.stem1 = ConvBNAct(3, mid_ch, 3, stride=2)
        self.stem2a = ConvBNAct(mid_ch, mid_ch // 2, 2, padding="same")
        self.stem2b = ConvBNAct(mid_ch // 2, mid_ch, 2, padding="same")
        self.stem3 = ConvBNAct(mid_ch * 2, mid_ch, 3, stride=2)
        self.stem4 = ConvBNAct(mid_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x2 = self.stem2b(self.stem2a(x))
        x1 = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=-float("inf")), kernel_size=2, stride=1)
        return self.stem4(self.stem3(torch.cat([x1, x2], dim=1)))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 4
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = self.fc2(scale)
        scale = torch.clamp((scale + 3.0) / 6.0, 0.0, 1.0)
        return x * scale


class RepDWConv(nn.Module):
    """Depthwise convolution with residual connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.branch3 = ConvBN(channels, channels, 3, groups=channels)
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
        self.identity_bn = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.branch3(x) + self.branch1(x) + x
        return self.identity_bn(y)


class LCNetV4Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int | Sequence[int],
        kernel: int,
        use_se: bool,
    ):
        super().__init__()
        self.stride = stride
        self.residual = isinstance(stride, int) and stride == 1 and in_ch == out_ch
        self.rep_dw = self.residual
        self.dw = RepDWConv(in_ch) if self.rep_dw else ConvBN(in_ch, in_ch, kernel, stride=stride, groups=in_ch)
        self.se = SqueezeExcite(in_ch) if use_se else None
        hidden = in_ch * 2
        self.expand = ConvBN(in_ch, hidden, 1)
        self.compress = ConvBN(hidden, out_ch, 1, bn_scale_init=0.0 if self.residual else 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        if self.se is not None:
            y = self.se(y)
        z = self.expand(y, activation="gelu")
        z = self.compress(z)
        return y + z if self.residual else z


def avg_pool_valid(x: torch.Tensor, kh: int = 3, kw: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    sh, sw = kh, kw
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    x = x[:, :, :oh * sh, :ow * sw].reshape(b, c, oh, kh, ow, kw)
    return x.mean(dim=(3, 5))


class PPLCNetV4Rec(nn.Module):
    def __init__(self, model_size: str = "medium"):
        super().__init__()
        if model_size not in REC_CONFIG:
            raise ValueError(f"Unknown PP-OCRv6 recognition size {model_size!r}")
        cfg = REC_CONFIG[model_size]
        mid, out = cfg["stem"]
        if cfg["stem_type"] == "branch":
            self.conv1 = StemBlock(mid, out)
        else:
            self.conv1a = ConvBN(3, mid, 3, stride=2)
            self.conv1b = ConvBN(mid, out, 3, stride=2)
        for stage in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            blocks = nn.ModuleList([
                LCNetV4Block(inc, outc, stride, kernel, se)
                for kernel, inc, outc, stride, se in cfg[stage]
            ])
            setattr(self, stage, blocks)
        self.model_size = model_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv1"):
            x = self.conv1(x)
        else:
            x = F.gelu(self.conv1a(x))
            x = F.gelu(self.conv1b(x))
        for stage in (self.blocks2, self.blocks3, self.blocks4, self.blocks5, self.blocks6):
            for block in stage:
                x = block(x)
        if self.training:
            return F.adaptive_avg_pool2d(x, (1, 40))
        return avg_pool_valid(x, 3, 2)


class ConvBNActSwish(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int | Sequence[int], groups: int = 1):
        super().__init__()
        self.block = ConvBN(in_ch, out_ch, kernel, groups=groups, use_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.block(x))


class AttentionBlock(nn.Module):
    """SVTR global-attention block with Paddle's prenorm=False ordering."""
    def __init__(
        self,
        dims: int,
        depth_index: int,
        mlp_ratio: float = 4.0,
        heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if dims % heads != 0:
            raise ValueError(f"dims {dims} must divide evenly across {heads} heads")
        self.heads = heads
        self.head_dim = dims // heads
        self.scale = self.head_dim ** -0.5
        self.depth_index = depth_index

        self.norm1 = nn.LayerNorm(dims, eps=1e-5)
        self.qkv = nn.Linear(dims, 3 * dims, bias=True)
        self.proj = nn.Linear(dims, dims)
        self.norm2 = nn.LayerNorm(dims, eps=1e-5)
        self.fc1 = nn.Linear(dims, int(dims * mlp_ratio))
        self.fc2 = nn.Linear(int(dims * mlp_ratio), dims)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        self.drop3 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        norm1_x = self.norm1(x.to(self.norm1.weight.dtype))
        qkv = self.qkv(norm1_x).reshape(b, t, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = F.softmax(attn.float(), dim=-1).to(v.dtype)
        attn = self.attn_dropout(attn)

        y = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.drop1(self.proj(y))
        y = F.silu(self.fc1(self.norm2(x.to(self.norm2.weight.dtype))))
        y = self.drop2(y)
        y = self.fc2(y)
        return x + self.drop3(y)


class LightSVTR(nn.Module):
    def __init__(
        self,
        in_ch: int = 768,
        dims: int = 192,
        depth: int = 2,
        local_kernel: int = 7,
    ):
        super().__init__()
        self.conv_reduce = ConvBNActSwish(in_ch, dims, 1)
        self.local = ConvBNActSwish(dims, dims, (1, local_kernel), groups=dims)
        self.blocks = nn.ModuleList([AttentionBlock(dims, i) for i in range(depth)])
        self.norm = nn.LayerNorm(dims, eps=1e-6)
        self.skip_conv = ConvBNActSwish(in_ch, dims, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(x)
        z = self.conv_reduce(x)
        z = z + self.local(z)
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z.to(self.norm.weight.dtype)).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return z + skip


class PP_OCRv6MediumRec(nn.Module):
    """PP-OCRv6 medium recognition CTC model in PyTorch."""

    def __init__(self, num_classes: int = 18710):
        super().__init__()
        self.backbone = PPLCNetV4Rec("medium")
        self.neck = LightSVTR(768, 192, 2, 7)
        self.ctc = nn.Linear(192, num_classes)

    def forward(self, images_nchw: torch.Tensor) -> torch.Tensor:
        x = self.backbone(images_nchw)
        x = self.neck(x)
        sequence = x[:, :, 0, :].permute(0, 2, 1)  # [B, 40, 192]
        logits = self.ctc(sequence)  # [B, 40, num_classes]
        return logits

    def forward_intermediates(self, images_nchw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named activations for exact numerical parity diagnosis."""
        values = {}
        x = self.backbone.conv1(images_nchw)
        values["backbone_stem"] = x
        for stage_name in ("blocks2", "blocks3", "blocks4", "blocks5", "blocks6"):
            blocks = getattr(self.backbone, stage_name)
            for block in blocks:
                x = block(x)
            if len(blocks) > 0:
                values[f"backbone_{stage_name}"] = x
        x = avg_pool_valid(x, 3, 2)
        values["backbone_pool"] = x

        skip = self.neck.skip_conv(x)
        values["neck_skip"] = skip
        z = self.neck.conv_reduce(x)
        values["neck_reduce"] = z
        local = self.neck.local(z)
        values["neck_local"] = local
        z = z + local
        values["neck_local_sum"] = z
        b, c, h, w = z.shape
        seq_z = z.permute(0, 2, 3, 1).reshape(b, h * w, c)
        for i, block in enumerate(self.neck.blocks):
            seq_z = block(seq_z)
            values[f"neck_block_{i}"] = seq_z
        norm_z = self.neck.norm(seq_z)
        values["neck_norm"] = norm_z
        z = norm_z.reshape(b, h, w, c).permute(0, 3, 1, 2) + skip
        values["neck_out"] = z
        sequence = z[:, :, 0, :].permute(0, 2, 1)
        values["sequence"] = sequence
        logits = self.ctc(sequence)
        values["ctc_logits"] = logits
        values["probabilities"] = F.softmax(logits.float(), dim=-1)
        return values
