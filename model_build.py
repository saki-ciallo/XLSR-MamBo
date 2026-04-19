import torch
import torch.nn as nn
import torch.nn.functional as F

import modules.step1_feature_ptm as step1
import modules.step2_model_blocks as step2
import modules.step3_pooling_outputs as step3

from dataclasses import dataclass, replace
from mamba_ssm import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, layer_norm_linear_fn


@dataclass
class Mamba2Config:
    # 通用配置，类名因早期开发固定，能用就行不修改了

    d_model: int = 128
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    
    mlp_type: str = "SwiGLU"
    
    mha_num_heads: int = 4
    mha_mlp_dim: int = 0
    mha_d_conv: int = 0
    
    gdn_num_heads: int = 4
    gdn_head_dim: int = 32
    
    
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

    use_mem_eff_path:  bool = False # Hydra 使用

    # process_group=None # 默认
    # 基本参数量：3 * expand * d_model^2
    def replace(self, **kwargs):
        return replace(self, **kwargs)


# 堆叠
class StackedMamba2(nn.Module):
    # todo 修改为只选择块类型
    def __init__(self,
                config: Mamba2Config,
                block_sequence: str,
                n_mamba: int,
                block_layers: int,
                block_cls: str,
                ):
        super().__init__()
        BLOCK_REGISTRY = {
            # "No1_Stack-Mamba": step2.Stack_Mamba2,
            # "No2_Mix-Mamba-GDN": step2.Mix_Mamba2_GDN, # GDN固定，支持 Mamba2 / Hydra
            # "No3_Mix-GDN-Mamba": step2.Mix_GDN_Mamba2, # 用于对比顺序不一致
            "No4_Mix-n_Mamba-1MLP": step2.Mix_n_Hydra_1MLP,
            "No5_Mix-n_Mamba-1MHA": step2.Mix_n_Hydra_1MHA,
            # "No6_Mix-n_Mamba-1MLP-1MHA": step2.Mix_n_Hydra_1MLP_1MHA,
            # "No7_Mix-n_Mamba-1MHA-1MLP": step2.Mix_n_Hydra_1MHA_1MLP,
            # "No8_Mix-1MHA-n_Mamba-1MLP": step2.Mix_1MHA_n_Hydra_1MLP,
            "No9_Mix-n_Mamba-1MLP-n_Mamba-1MHA": step2.Mix_n_Hydra_1MLP_n_Hydra_1MHA,
            # "No10_Mix-n_Mamba-1MHA-n_Mamba-1MLP": step2.Mix_n_Hydra_1MHA_n_Hydra_1MLP,
            # "No11_Mix-1MHA-n_Mamba-1MLP-n_Mamba": step2.Mix_1MHA_n_Hydra_1MLP_n_Hydra,
            "No12_Mix_n_Mamba_1MLP_1MHA_1MLP": step2.Mix_n_Hydra_1MLP_1MHA_1MLP,
            # "No13_Mix-MM-MM": step2.Mix_Mamba2_MLP_MHA_MLP,
        }

        # 在内部已修改为 Add -> LN -> Attn / MLP / Mixer 结构
        block_class = BLOCK_REGISTRY.get(block_sequence)
        if block_class is None:
            raise ValueError(f"Error : Unknown block_type '{block_sequence}'。 "
                                f"The valid keys are: {BLOCK_REGISTRY.keys()}")
        else:
            self.layers = block_class(config, block_layers, n_mamba, block_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    

class F4_XLSR300M_General(nn.Module):
    def __init__(self, model_config: dict, freeze: bool, block_layers: int, 
                 block_sequence: str, n_mamba: int, block_cls: str, fixed_length_samples: int):
        super().__init__()
        self.mamba_config = model_config["mamba_config"]
        self.block_layers = block_layers
        self.block_sequence = block_sequence
        self.n_mamba = n_mamba

        # 1. 特征提取器
        self.feature_extractor = step1.WLSR300M_Extractor(freeze=freeze)
        
        # 2. 输入投影
        self.input_proj = step1.RMS_InputProj(self.mamba_config.d_model, dropout=0.0) # batch不敏感
        
        # 3. Mamba2 堆叠模块
        self.stacked_mamba2 = StackedMamba2(config=self.mamba_config,
                                            block_layers=self.block_layers,
                                            block_sequence=self.block_sequence,
                                            n_mamba=self.n_mamba,
                                            block_cls=block_cls,
                                            )
        
        # 4. 注意力池化
        self.attention_pool = step3.GatedAttentionPool(self.mamba_config.d_model)
        
        # 5. 简单分类头
        self.classifier = nn.Linear(self.mamba_config.d_model, 2)
        
        self._init_input_proj()
        
    def _init_input_proj(self):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        # x = self.toy_extractor(x) # 将 (B, L) -> (B, L, 1) 以适配线性层输入
        x = self.input_proj(x)
        x = self.stacked_mamba2(x)
        x = self.attention_pool(x)
        x = self.classifier(x)
        return x


class F5_WAVLM_LARGE_General(nn.Module):
    def __init__(self, model_config: dict, freeze: bool, block_layers: int, 
                 block_sequence: str, n_mamba: int, block_cls: str, fixed_length_samples: int):
        super().__init__()
        self.mamba_config = model_config["mamba_config"]
        self.block_layers = block_layers
        self.block_sequence = block_sequence
        self.n_mamba = n_mamba
        
        # 1. 特征提取器
        self.feature_extractor = step1.WaveLM_Large_Extractor(freeze=freeze)
        
        # 2. 输入投影
        self.input_proj = step1.RMS_InputProj(self.mamba_config.d_model, dropout=0.0) # batch不敏感
        
        # 3. Mamba2 堆叠模块
        self.stacked_mamba2 = StackedMamba2(config=self.mamba_config,
                                            block_layers=self.block_layers,
                                            block_sequence=self.block_sequence,
                                            n_mamba=self.n_mamba,
                                            block_cls=block_cls,
                                            )
        
        # 4. 注意力池化
        self.attention_pool = step3.GatedAttentionPool(self.mamba_config.d_model)
        
        # 5. 简单分类头
        self.classifier = nn.Linear(self.mamba_config.d_model, 2)
        
        self._init_input_proj()
        
    def _init_input_proj(self):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.input_proj(x)
        x = self.stacked_mamba2(x)
        x = self.attention_pool(x)
        x = self.classifier(x)
        return x

class F6_WAV2VEC2_LARGE_General(nn.Module):
    def __init__(self, model_config: dict, freeze: bool, block_layers: int, 
                 block_sequence: str, n_mamba: int, block_cls: str, fixed_length_samples: int):
        super().__init__()
        self.mamba_config = model_config["mamba_config"]
        self.block_layers = block_layers
        self.block_sequence = block_sequence
        self.n_mamba = n_mamba
        
        # 1. 特征提取器
        self.feature_extractor = step1.Wav2Vec2_Large_Extractor(freeze=freeze)
        
        # 2. 输入投影
        self.input_proj = step1.RMS_InputProj(self.mamba_config.d_model, dropout=0.0) # batch不敏感
        
        # 3. Mamba2 堆叠模块
        self.stacked_mamba2 = StackedMamba2(config=self.mamba_config,
                                            block_layers=self.block_layers,
                                            block_sequence=self.block_sequence,
                                            n_mamba=self.n_mamba,
                                            block_cls=block_cls,
                                            )
        
        # 4. 注意力池化
        self.attention_pool = step3.GatedAttentionPool(self.mamba_config.d_model)
        
        # 5. 简单分类头
        self.classifier = nn.Linear(self.mamba_config.d_model, 2)
        
        self._init_input_proj()
        
    def _init_input_proj(self):
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.input_proj(x)
        x = self.stacked_mamba2(x)
        x = self.attention_pool(x)
        x = self.classifier(x)
        return x
