import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from .hydra import Hydra
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.modules.mha import MHA
from fla.layers import GatedDeltaNet

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    


@dataclass
class Mamba2Config:
    d_model: int = 128
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 32
    
    mlp_type: str = "SwiGLU"
    
    mha_num_heads: int = 4
    mha_mlp_dim: int = 0
    mha_d_conv: int = 4
    
    gdn_num_heads: int = 4
    gdn_head_dim: int = 32
        
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

    use_mem_eff_path:  bool = False
    
    
    def replace(self, **kwargs):
        return replace(self, **kwargs)
    
    
    


# 参考 PackedSwiGLUMLP 设计，相比较不拆分的方法有明显加速
# https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
class MLP_SwiGLU(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        """
        (Swish1(xW)⊗xV)W2
        内存里W和V拼接形成一个大权重矩阵 W = [w|v]
        那么 combined = xW = x[w|v] = [xw | xv]
        这种方法可以减少一次矩阵乘，只需要拆分即可得到 gate 和 value
        """
        hidden_dim = int(8 * d_model / 3)
        self.w1_w3_combined = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.w2_out = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.SiLU() # beta=1 的 Swish 激活函数
        
        nn.init.kaiming_normal_(self.w1_w3_combined.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.w2_out.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        # x shape: (B, L, d_model)
        combined = self.w1_w3_combined(x)
        gate, value = combined.chunk(2, dim=-1)
        x = self.w2_out(self.activation(gate) * value)
        x = self.dropout(x)
        return x

class MLP_SiLU(nn.Module):
    """
    简单 SiLU MLP 实现
    """
    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = int(8 * d_model / 3)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


######## 独立块构建，均采用 Add -> LN -> Mixer 结构 ########
class MLP_block(nn.Module):
    """
    可选 SiLU 或 SwiGLU 的 Pre-LN MLP 块
    Add -> LN -> Mixer 结构，返回 hidden_states 和 residual
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        if config.mlp_type == "SiLU":
            self.mlp = MLP_SiLU(config.d_model)
        elif config.mlp_type == "SwiGLU":
            self.mlp = MLP_SwiGLU(config.d_model, dropout=0.1)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        
    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual

class Hydra_Block(nn.Module):
    """
    独立 Hydra 块，返回 hidden_states 和 residual
    [Add, LN, Hydra]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model) # 只使用 RMSNorm，不额外定义 LayerNorm
        self.mamba = Hydra(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv, # 默认 7
            expand=config.expand,
            use_mem_eff_path=config.use_mem_eff_path,
        )
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
            
    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        hidden_states = self.mamba(hidden_states)

        return hidden_states, residual

class Mamba1_Block(nn.Module):
    """
    独立 Mamba1 块，返回 hidden_states 和 residual
    [Add, LN, Mamba1]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model) # 只使用 RMSNorm，不额外定义 LayerNorm
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state, # 仅支持16
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
            
    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        hidden_states = self.mamba(hidden_states)

        return hidden_states, residual



class Mamba2_Block(nn.Module):
    """
    独立 Mamba2 块，返回 hidden_states 和 residual
    [Add, LN, Mamba2]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model) # 只使用 RMSNorm，不额外定义 LayerNorm
        self.mamba = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
            
    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        hidden_states = self.mamba(hidden_states)

        return hidden_states, residual
    

class BiMamba_add_Block(nn.Module):
    """
    Bi-Mamba 块，不共享 Mamba 参数，返回 hidden_states 和 residual
    [Add, Share_Norm, BiMamba_add]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model) # 共享 Norm
        self.mamba_f = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.mamba_b = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm


    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        f_hidden_states = hidden_states
        b_hidden_states = hidden_states.flip([1])
        y_fwd = self.mamba_f(f_hidden_states)
        y_bwd = self.mamba_b(b_hidden_states)
        hidden_states = y_fwd + y_bwd.flip([1])
        return hidden_states, residual


class BiMamba2_add_Block(nn.Module):
    """
    Bi-Mamba2 块，不共享 Mamba2 参数，返回 hidden_states 和 residual
    [Add, Share_Norm, BiMamba_add]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model) # 共享 Norm
        self.mamba2_f = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.mamba2_b = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm


    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        f_hidden_states = hidden_states
        b_hidden_states = hidden_states.flip([1])
        y_fwd = self.mamba2_f(f_hidden_states)
        y_bwd = self.mamba2_b(b_hidden_states)
        hidden_states = y_fwd + y_bwd.flip([1])
        return hidden_states, residual

class BiMamba2_cat_Block(nn.Module):
    """
    # 参考 XLSR-Mamba 合并和映射方法
    Bi-Mamba2 块，不共享 Mamba2 参数，返回 hidden_states 和 residual
    [Add, Share_Norm, BiMamba_cat, Linear]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model) # 共享 Norm
        self.mamba2_f = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.mamba2_b = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.linear = nn.Linear(2 * config.d_model, config.d_model)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        f_hidden_states = hidden_states
        b_hidden_states = hidden_states.flip([1])
        f_hidden_states = self.mamba2_f(f_hidden_states)
        b_hidden_states = self.mamba2_b(b_hidden_states)
        hidden_states = torch.cat((f_hidden_states, b_hidden_states), dim=-1)
        hidden_states = self.linear(hidden_states)
        return hidden_states, residual
    
class MHA_Block(nn.Module):
    """
    独立 Multi-Head Attention 块，返回 hidden_states 和 residual
    [Add, Norm, MHA]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mha = MHA(
            embed_dim=config.d_model,
            num_heads=config.mha_num_heads,
            mlp_dim=config.mha_mlp_dim,
            d_conv=config.mha_d_conv
            )
        self.residual_in_fp32 = config.residual_in_fp32 
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        hidden_states = self.mha(hidden_states)

        return hidden_states, residual
    
class MHA_rope_Block(nn.Module):
    """
    开启 RoPE 的独立 Multi-Head Attention 块，返回 hidden_states 和 residual
    [Add, Norm, MHA]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mha = MHA(
            embed_dim=config.d_model,
            num_heads=config.mha_num_heads,
            mlp_dim=config.mha_mlp_dim,
            d_conv=config.mha_d_conv,
            rotary_emb_dim=config.d_model // config.mha_num_heads,
            )
        self.residual_in_fp32 = config.residual_in_fp32 
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # B, L, D = hidden_states.shape
        # position_ids = torch.arange(L, device=hidden_states.device, dtype=torch.long)
        # position_ids = position_ids.unsqueeze(0).expand(B, L)
        hidden_states = self.mha(hidden_states)

        return hidden_states, residual

    
class GDN_Block(nn.Module):
    """
    独立 FLA Gated DeltaNet 块，返回 hidden_states 和 residual
    [Add, Norm, GatedDeltaNet]
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.gdn = GatedDeltaNet(
            hidden_size=config.d_model,
            head_dim=config.gdn_head_dim,
            num_heads=config.gdn_num_heads # 其它参数保持默认
            )
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # Norm
        # hidden_states = self.norm(residual)
        # Mixer
        hidden_states, _, _ = self.gdn(hidden_states) # 不需要使用 past_key_values

        return hidden_states, residual


######## 紧密混合双块构建，均采用 Add -> LN -> Mixer 结构 ########
# class N_GDN_MLP_Block(nn.Module):
#     """
#     紧密 GatedDeltaNet - MLP 块，返回 hidden_states 和 residual
#     [Add, Norm, GDN, MLP]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model)
#         self.gdn = GatedDeltaNet(
#             hidden_size=config.d_model,
#             head_dim=config.gdn_head_dim,
#             num_heads=config.gdn_num_heads # 其它参数保持默认
#             )
#         self.mlp = MLP_block(config)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             hidden_states, residual = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=True,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         # Norm
#         # hidden_states = self.norm(residual)
#         # Mixer
#         hidden_states, _, _ = self.gdn(hidden_states)
#         hidden_states = self.mlp(hidden_states)

#         return hidden_states, residual

# class N_Mamba2_MLP_Block(nn.Module):
#     """
#     紧密 Mamba2 - MLP 块，返回 hidden_states 和 residual
#     [Add, Norm, Mamba2, MLP]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model) # 只使用 RMSNorm，不额外定义 LayerNorm
#         self.mamba = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.mlp = MLP_block(config)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm
            
#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             hidden_states, residual = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=True,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         # Norm
#         # hidden_states = self.norm(residual)
#         # Mixer
#         hidden_states = self.mamba(hidden_states)
#         hidden_states = self.mlp(hidden_states)

#         return hidden_states, residual


# class N_BiMamba2_add_MLP_Block(nn.Module):
#     """
#     紧密 Mamba2_Add - MLP 块，不共享参数，返回 hidden_states 和 residual
#     [Add, Norm, BiMamba2_add, MLP]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model) # 共享 Norm
#         self.mamba2_f = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.mamba2_b = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.mlp = MLP_block(config)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             hidden_states, residual = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=True,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         # Norm
#         # hidden_states = self.norm(residual)
#         # Mixer
#         f_hidden_states = hidden_states
#         b_hidden_states = hidden_states.flip([1])
#         y_fwd = self.mamba2_f(f_hidden_states)
#         y_bwd = self.mamba2_b(b_hidden_states)
#         hidden_states = y_fwd + y_bwd.flip([1])
#         hidden_states = self.mlp(hidden_states)
        
#         return hidden_states, residual

# class N_BiMamba2_cat_MLP_Block(nn.Module):
#     """
#     # 参考 XLSR-Mamba 合并和映射方法
#     紧密 BiMamba2_cat - MLP 块，不共享参数，返回 hidden_states 和 residual
#     [Add, Share_Norm, BiMamba_cat, Linear]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model) # 共享 Norm
#         self.mamba2_f = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.mamba2_b = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.linear = nn.Linear(2 * config.d_model, config.d_model)
#         self.mlp = MLP_block(config)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             hidden_states, residual = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=True,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         # Norm
#         # hidden_states = self.norm(residual)
#         # Mixer
#         f_hidden_states = hidden_states
#         b_hidden_states = hidden_states.flip([1])
#         f_hidden_states = self.mamba2_f(f_hidden_states)
#         b_hidden_states = self.mamba2_b(b_hidden_states)
#         hidden_states = torch.cat((f_hidden_states, b_hidden_states), dim=-1)
#         hidden_states = self.linear(hidden_states)
#         hidden_states = self.mlp(hidden_states)
        
#         return hidden_states, residual

# class N_MHA_MLP_Block(nn.Module):
#     """
#     紧密 MHA - MLP 块，返回 hidden_states 和 residual
#     [Add, Norm, Multi-Head Attention, MLP]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model)
#         self.mha= MHA(
#             embed_dim=config.d_model,
#             num_heads=config.mha_num_heads,
#             mlp_dim=config.mha_mlp_dim,
#             d_conv=config.mha_d_conv
#             )
#         self.mlp = MLP_block(config)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             hidden_states, residual = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=True,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         # Norm
#         # hidden_states = self.norm(residual)
#         # Mixer
#         hidden_states = self.mha(hidden_states)
#         hidden_states = self.mlp(hidden_states)
        
#         return hidden_states, residual
    
# class Bi_Add_Mamba2_Block_old(nn.Module):
#     """
#     双向加 Mamba2 块，包含残差连接
#     [Norm, (Mamba2_Fwd, Mamba2_Bwd), (Add_Fwd, Add_Bwd), Merge + Residual_Add]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
        
#         self.mamba2_fwd = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.mamba2_bwd = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.norm1 = RMSNorm(config.d_model) # 共享 Norm

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Forward pass
#         y_fwd = self.mamba2_fwd(self.norm1(x))
#         # Backward pass: 翻转序列，应用 mamba2，再翻转回
#         y_bwd = self.mamba2_bwd(self.norm1(x.flip(1)))
#         # 合并前向和后向的输出
#         x_merged = y_fwd + y_bwd.flip(1)
#         return x_merged + x


################ 简单块构建 ################
class Stack_Mamba2(nn.Module):
    """
    重复堆叠多个 Mamba2 块
    支持选择 Mamba2 / Bi_add / Bi_cat / Hydra / GDN
    [Mamba2]
    """
    def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(block_layers):
            # 添加 Mamba 层
            if block_cls == "Bi2_add":
                mamba_layer = BiMamba2_add_Block
            elif block_cls == "Bi2_cat":
                mamba_layer = BiMamba2_cat_Block
            elif block_cls == "Bi_add":
                mamba_layer = BiMamba_add_Block
            elif block_cls == "Mamba2":
                mamba_layer = Mamba2_Block
            elif block_cls == "Mamba1":
                mamba_layer = Mamba1_Block
            elif block_cls == "Hydra":
                mamba_layer = Hydra_Block
            elif block_cls == "GDN":
                mamba_layer = GDN_Block
            else:
                raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
            self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            
        self.norm = RMSNorm(config.d_model)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 第一个块可以处理 residual = None 的情况，无需像独立块那样每次都判断
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        # 后续传递给下一个模块，不再需要返回 residual
        return hidden_states
    

# class Mix_Mamba2_GDN(nn.Module):
#     """
#     混合散装 Mamba2 - GDN 块
#     支持选择 Mamba2 / Bi_add / Bi_cat / Hydra 的混合SSM
#     [Mamba2, GDN]
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             # 添加 Mamba 层
#             if block_cls == "Bi_add":
#                 mamba_layer = BiMamba2_add_Block
#             elif block_cls == "Bi_cat":
#                 mamba_layer = BiMamba2_cat_Block
#             elif block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
            
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            
#             # 添加 GDN 层
#             self.layers.append(GDN_Block(config))
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # 第一个块可以处理 residual = None 的情况，无需像独立块那样每次都判断
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
        
#         # if residual is not None:
#         #     hidden_states = hidden_states + residual
#         # hidden_states = self.norm(hidden_states)
#         # 后续传递给下一个模块，不再需要返回 residual
#         return hidden_states
    
# class Mix_GDN_Mamba2(nn.Module):
#     """
#     混合散装 GDN - Mamba2 块
#     支持选择 Mamba2 / Bi_add / Bi_cat / Hydra 的混合SSM
#     [GDN, Mamba2], 顺序与 Mix_Mamba2_GDN 相反
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             # 添加 GDN 层
#             self.layers.append(GDN_Block(config))
            
#             if block_cls == "Bi_add":
#                 mamba_layer = BiMamba2_add_Block
#             elif block_cls == "Bi_cat":
#                 mamba_layer = BiMamba2_cat_Block
#             elif block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#             # 添加 Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # 第一个块可以处理 residual = None 的情况，无需像独立块那样每次都判断
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
        
#         # if residual is not None:
#         #     hidden_states = hidden_states + residual
#         # hidden_states = self.norm(hidden_states)
#         # 后续传递给下一个模块，不再需要返回 residual
#         return hidden_states
    
class Mix_n_Hydra_1MLP(nn.Module):
    """
    支持 Mamba2 / Hydra / GDN
    3 个 Hydra - 1 个 MLP 块
    """
    def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(block_layers):
            if block_cls == "Mamba2":
                mamba_layer = Mamba2_Block
            elif block_cls == "Mamba1":
                mamba_layer = Mamba1_Block
            elif block_cls == "Hydra":
                mamba_layer = Hydra_Block
            elif block_cls == "GDN":
                mamba_layer = GDN_Block
            elif block_cls == "Bi_add":
                mamba_layer = BiMamba_add_Block
            else:
                raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
            # 添加 Mamba 层
            self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            # 添加 MLP 层
            self.layers.append(MLP_block(config))
            
        self.norm = RMSNorm(config.d_model)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # Use long-range features as the key for residual connections
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        return hidden_states
    
class Mix_n_Hydra_1MHA(nn.Module):
    """
    支持 Mamba2 / Hydra / GDN
    3 个 Hydra - 1 个 MHA 块
    """
    def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(block_layers):
            if block_cls == "Mamba2":
                mamba_layer = Mamba2_Block
            elif block_cls == "Mamba1":
                mamba_layer = Mamba1_Block
            elif block_cls == "Hydra":
                mamba_layer = Hydra_Block
            elif block_cls == "GDN":
                mamba_layer = GDN_Block
            elif block_cls == "Bi_add":
                mamba_layer = BiMamba_add_Block
            else:
                raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
            # 添加 Mamba 层
            self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            # 添加 MHA 层
            self.layers.append(MHA_Block(config))
            
        self.norm = RMSNorm(config.d_model)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # Use long-range features as the key for residual connections
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        return hidden_states


# class Mix_1MHA_n_Hydra_1MLP(nn.Module):
#     """
#     支持 Mamba2 / Hydra / GDN
#     1 个 MHA - 3 个 Hydra - 1 个 MLP 块
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             # 添加 MHA 层
#             self.layers.append(MHA_Block(config))
            
#             # 添加 Mamba 层
#             if block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             elif block_cls == "GDN":
#                 mamba_layer = GDN_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#             # 添加 Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
#             # 添加 MLP 层
#             self.layers.append(MLP_block(config))
            
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         return hidden_states
    
# class Mix_n_Hydra_1MLP_1MHA(nn.Module):
#     """
#     支持 Mamba2 / Hydra / GDN
#     3 个 Hydra - 1 个 MLP - 1 个 MHA 块
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             if block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             elif block_cls == "GDN":
#                 mamba_layer = GDN_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#             # 添加 Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            
#             # 添加 MLP 层
#             self.layers.append(MLP_block(config))
            
#             # 添加 MHA 层
#             self.layers.append(MHA_Block(config))
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         return hidden_states
    
# class Mix_n_Hydra_1MHA_1MLP(nn.Module):
#     """
#     支持 Mamba2 / Hydra / GDN
#     3 个 Hydra - 1 个 MHA - 1 个 MLP 块
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             if block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             elif block_cls == "GDN":
#                 mamba_layer = GDN_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#             # 添加 Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            
#             # 添加 MHA 层
#             self.layers.append(MHA_Block(config))
            
#             # 添加 MLP 层
#             self.layers.append(MLP_block(config))
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         return hidden_states



class Mix_n_Hydra_1MLP_n_Hydra_1MHA(nn.Module):
    """
    支持 Mamba2 / Hydra / GDN
    3 个 Hydra - 1 个 MLP - 3 个 Hydra - 1 个 MHA 块
    """
    def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(block_layers):
            if block_cls == "Mamba2":
                mamba_layer = Mamba2_Block
            elif block_cls == "Mamba1":
                mamba_layer = Mamba1_Block
            elif block_cls == "Hydra":
                mamba_layer = Hydra_Block
            elif block_cls == "GDN":
                mamba_layer = GDN_Block
            elif block_cls == "Bi_add":
                mamba_layer = BiMamba_add_Block
            else:
                raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
            # 添加前x Mamba 层
            self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            # 添加 MLP 层
            self.layers.append(MLP_block(config))
            
            # 添加后x Mamba 层
            self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            # 添加 MHA 层
            self.layers.append(MHA_Block(config))
            
        self.norm = RMSNorm(config.d_model)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # Use long-range features as the key for residual connections
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        return hidden_states
    
# class Mix_n_Hydra_1MHA_n_Hydra_1MLP(nn.Module):
#     """
#     支持 Mamba2 / Hydra / GDN
#     3 个 Hydra - 1 个 MHA - 3 个 Hydra - 1 个 MLP 块
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             if block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             elif block_cls == "GDN":
#                 mamba_layer = GDN_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#             # 添加前x Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
#             # 添加 MHA 层
#             self.layers.append(MHA_Block(config))
            
#             # 添加后x Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
#             # 添加 MLP 层
#             self.layers.append(MLP_block(config))
            
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         return hidden_states


# class Mix_1MHA_n_Hydra_1MLP_n_Hydra(nn.Module):
#     """
#     支持 Hydra / Single
#     1 个 MHA - 3 个 Hydra - 1 个 MLP - 3 个 Hydra 块
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
#             if block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             elif block_cls == "GDN":
#                 mamba_layer = GDN_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#              # 添加 MHA 层
#             self.layers.append(MHA_Block(config))
#             # 添加前x Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            
#             # 添加 MLP 层
#             self.layers.append(MLP_block(config))
#             # 添加后x Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
                        
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
#         return hidden_states
    
class Mix_n_Hydra_1MLP_1MHA_1MLP(nn.Module):
    """
    支持 Hydra / Single
    3 个 Hydra - 1 个 MLP - 1 个 MHA - 1 个 MLP 块
    """
    def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(block_layers):
            if block_cls == "Mamba2":
                mamba_layer = Mamba2_Block
            elif block_cls == "Mamba1":
                mamba_layer = Mamba1_Block
            elif block_cls == "Hydra":
                mamba_layer = Hydra_Block
            elif block_cls == "GDN":
                mamba_layer = GDN_Block
            elif block_cls == "Bi_add":
                mamba_layer = BiMamba_add_Block
            else:
                raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
            # 添加前x Mamba 层
            self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
            # 添加 MLP1 层
            self.layers.append(MLP_block(config))
            
            # 添加 MHA 层
            self.layers.append(MHA_Block(config))
            # 添加 MLP2 层
            self.layers.append(MLP_block(config))
                        
        self.norm = RMSNorm(config.d_model)
        
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Add & Norm
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # Use long-range features as the key for residual connections
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        return hidden_states



# class Mix_Mamba2_MLP_MHA_MLP(nn.Module):
#     """
#     混合散装 Mamba2 - MLP1 - MHA - MLP2 块
#     支持选择 Mamba2 / Bi_add / Bi_cat / Hydra / GDN 的混合SSM
#     [Mamba2, GDN]
#     """
#     def __init__(self, config: Mamba2Config, block_layers: int, n_mamba: int, block_cls: str):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(block_layers):
            
#             if block_cls == "Bi_add":
#                 mamba_layer = BiMamba2_add_Block
#             elif block_cls == "Bi_cat":
#                 mamba_layer = BiMamba2_cat_Block
#             elif block_cls == "Mamba2":
#                 mamba_layer = Mamba2_Block
#             elif block_cls == "Mamba1":
#                 mamba_layer = Mamba1_Block
#             elif block_cls == "Hydra":
#                 mamba_layer = Hydra_Block
#             elif block_cls == "GDN":
#                 mamba_layer = GDN_Block
#             else:
#                 raise ValueError(f"Unsupported mixer_mamba_type: {block_cls}")
            
#             # 添加 Mamba 层
#             self.layers.extend([mamba_layer(config) for _ in range(n_mamba)])
#             # 添加 MLP1
#             self.layers.append(MLP_block(config))
#             # 添加 MHA 层
#             self.layers.append(MHA_Block(config))
#             # 添加 MLP2
#             self.layers.append(MLP_block(config))
            
#         self.norm = RMSNorm(config.d_model)
        
#         self.residual_in_fp32 = config.residual_in_fp32
#         self.fused_add_norm = config.fused_add_norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # 第一个块可以处理 residual = None 的情况，无需像独立块那样每次都判断
#         for layer in self.layers:
#             hidden_states, residual = layer(hidden_states, residual)
        
#         # Add & Norm
#         if not self.fused_add_norm:
#             residual = (hidden_states + residual) if residual is not None else hidden_states
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             # Use long-range features as the key for residual connections
#             hidden_states = fused_add_norm_fn(
#                 hidden_states,
#                 self.norm.weight,
#                 self.norm.bias,
#                 residual=residual,
#                 prenorm=False,
#                 residual_in_fp32=self.residual_in_fp32,
#                 eps=self.norm.eps,
#             )
        
#         return hidden_states



# 以下内容为早期开发使用

# # 双向设计 Bi-PreLN-Mamba2 结构
# class PreLN_Bi_MLP(nn.Module):
#     """
#     参考 transformer Pre-LN 设计的 Mamba2 块
#     [(RMSNorm1, Bi-Mamba2, Residual_Add1), (RMSNorm2, MLP, Residual_Add2)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.merged = Bi_Add_Mamba2_Block(config)
#         self.MLP = MLP_block(config.mlp_type, config.d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mamba_out = self.merged(x)
#         return self.MLP(mamba_out)



# #------ 混合 MHA + Mamba 结构 ------#
# class MHA_Single_Structure(nn.Module):
#     """
#     参考 https://openreview.net/pdf?id=GbFluKMmtE 实现 MambaFormer
#     [(MHA, Residual_Add), (Single-Mamba2, Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.mha= MHA(
#             embed_dim=config.d_model,
#             num_heads=config.mha_num_heads,
#             mlp_dim=config.mha_mlp_dim, #如果启用则gated结构
#             d_conv=config.mha_d_conv
#             )
#         self.mamba = Single_Mamba2_Block(config) # 包含残差连接

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mha_out = self.mha(x) + x # MHA 残差连接
#         mamba_out = self.mamba(mha_out)
#         return mamba_out

# class MHA_Bi_Structure(nn.Module):
#     """
#     [(MHA, Residual_Add), (Bi-Mamba2, Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.mha= MHA(
#             embed_dim=config.d_model,
#             num_heads=config.mha_num_heads,
#             mlp_dim=config.mha_mlp_dim, #如果启用则gated结构
#             d_conv=config.mha_d_conv
#             )
#         self.mamba = Bi_Add_Mamba2_Block(config) # 包含残差连接

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mha_out = self.mha(x) + x # MHA 残差连接
#         mamba_out = self.mamba(mha_out)
#         return mamba_out
    
# # [Mamba + MHA] 结构
# class Single_MHA_Structure(nn.Module):
#     """
#     [(Single-Mamba2, Residual_Add), (MHA, Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.mha= MHA(
#             embed_dim=config.d_model,
#             num_heads=config.mha_num_heads,
#             mlp_dim=config.mha_mlp_dim, #如果启用则gated结构
#             d_conv=config.mha_d_conv
#             )
#         self.mamba = Single_Mamba2_Block(config) # 包含残差连接

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mamba_out = self.mamba(x)
#         mha_out = self.mha(mamba_out) + mamba_out # MHA 残差连接
#         return mha_out
    
# class Bi_MHA_Structure(nn.Module):
#     """
#     [(Bi-Mamba2, Residual_Add), (MHA, Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.mha= MHA(
#             embed_dim=config.d_model,
#             num_heads=config.mha_num_heads,
#             mlp_dim=config.mha_mlp_dim, #如果启用则gated结构
#             d_conv=config.mha_d_conv
#             )
#         self.mamba = Bi_Add_Mamba2_Block(config) # 包含残差连接

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mamba_out = self.mamba(x)
#         mha_out = self.mha(mamba_out) + mamba_out # MHA 残差连接
#         return mha_out
    
    
    
    
# # 参考 Gated DeltaNet 设计
# class Res_Single_MLP_Structure(nn.Module):
#     """
#     紧密 Single-Mamba2 + MLP 结构
#     [(RMSNorm, Single-Mamba2, MLP), Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model)
#         self.mamba = Mamba2(
#             d_model=config.d_model,
#             d_state=config.d_state,
#             d_conv=config.d_conv,
#             expand=config.expand
#         )
#         self.MLP = MLP_SwiGLU(config.d_model, dropout=0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mamba_out = self.mamba(self.norm(x))
#         MLP_out = self.MLP(mamba_out) + x
#         return MLP_out

# class Res_Single_MLP_Structure_new(nn.Module):
#     """
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model)
#         self.blocks = nn.ModuleList()
#         for _ in range(config.block_layers):
#             self.blocks.append(Single_Mamba2_Block(config)) # 已包含 norm
#             self.blocks.append(MLP_block(config.d_model, dropout=0)) # 已包含 norm

#     def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        
#         residual = None # 初始状态
#         for block in self.blocks:
#             hidden_states, residual = block(hidden_states, residual)
#         hidden_states = self.norm(hidden_states + residual) # residual 一定不为空
#         return hidden_states


# class Res_Bi_MLP_Structure(nn.Module):
#     """
#     紧密 Bi-Mamba2 + MLP 结构
#     [(RMSNorm, Bi-Mamba2, Residual_Add), MLP, Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.mamba = Bi_Add_Mamba2_Block(config)
#         self.MLP = MLP_SwiGLU(config.d_model, dropout=0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mamba_out = self.mamba(x)
#         MLP_out = self.MLP(mamba_out) + x
#         return MLP_out

# class Res_MHA_MLP_Structure(nn.Module):
#     """
#     紧密 MHA + MLP 结构
#     [(RMSNorm, MHA, MLP), Residual_Add)]
#     """
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.norm = RMSNorm(config.d_model)
#         self.mha= MHA(
#             embed_dim=config.d_model,
#             num_heads=config.mha_num_heads,
#             mlp_dim=config.mha_mlp_dim,
#             d_conv=config.mha_d_conv
#             )
#         self.MLP = MLP_SwiGLU(config.d_model, dropout=0)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         mha_out = self.mha(self.norm(x))
#         MLP_out = self.MLP(mha_out) + x
#         return MLP_out

# class Hybrid_SF_MF(nn.Module):
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.sf = Res_Single_MLP_Structure(config)
#         self.mf = Res_MHA_MLP_Structure(config)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.mf(self.sf(x))
    
# class Hybrid_BF_MF(nn.Module):
#     def __init__(self, config: Mamba2Config):
#         super().__init__()
#         self.bf = Res_Bi_MLP_Structure(config)
#         self.mf = Res_MHA_MLP_Structure(config)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.mf(self.bf(x))