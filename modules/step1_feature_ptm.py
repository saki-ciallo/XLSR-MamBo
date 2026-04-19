import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
from dataclasses import dataclass
from transformers import Wav2Vec2Model, WavLMModel
import torchaudio
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
    

class WLSR300M_Extractor(nn.Module):
    def __init__(self, 
                 model_bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M,
                 freeze: bool = True
                 ):
        super().__init__()
        
        # torchaudio
        self.model = model_bundle.get_model()
        self.freeze = freeze
        
        if freeze:
            # for param in self.model.parameters():
            #     param.requires_grad = False
            self.model.requires_grad_(False)
        else:
            self.model.requires_grad_(True)
            self.model.train()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform, shape (B, T): (B, 64000)
        Output: features, shape (B, L, D): (B, 199, 1024)
        """
        if self.freeze:
            with torch.no_grad():
                # torchaudio
                outputs, _ = self.model(waveform)
                return outputs
        else:
            # torchaudio
            outputs, _ = self.model(waveform)
            return outputs


class WaveLM_Large_Extractor(nn.Module):
    def __init__(self, 
                 model_bundle = torchaudio.pipelines.WAVLM_LARGE,
                 freeze: bool = True
                 ):
        super().__init__()
        
        self.model = model_bundle.get_model()
        self.freeze = freeze
        
        if freeze:
            self.model.requires_grad_(False)
        else:
            self.model.requires_grad_(True)
            self.model.train()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform, shape (B, T): (B, 64000)
        Output: features, shape (B, L, D): (B, 199, 1024)
        """
        
        if self.freeze:
            with torch.no_grad():
                # torchaudio
                outputs, _ = self.model(waveform)
                return outputs
        else:
            # torchaudio
            outputs, _ = self.model(waveform)
            return outputs
        

class Wav2Vec2_Large_Extractor(nn.Module):
    def __init__(self, 
                 model_bundle = torchaudio.pipelines.WAV2VEC2_LARGE,
                 freeze: bool = True
                 ):
        super().__init__()
        
        self.model = model_bundle.get_model()
        self.freeze = freeze
        
        if freeze:
            self.model.requires_grad_(False)
        else:
            self.model.requires_grad_(True)
            self.model.train()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform, shape (B, T): (B, 64000)
        Output: features, shape (B, L, D): (B, 199, 1024)
        """
        
        if self.freeze:
            with torch.no_grad():
                # torchaudio
                outputs, _ = self.model(waveform)
                return outputs
        else:
            # torchaudio
            outputs, _ = self.model(waveform)
            return outputs

        

class BN_InputProj(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1024, d_model, dropout)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        x = self.dropout(x)
        return x

class LN_InputProj(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1024, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        return x
    
class RMS_InputProj(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1024, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.dropout(x)
        return x