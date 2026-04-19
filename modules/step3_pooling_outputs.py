import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm


# Global Pooling: GlobalAvgPool + GlobalMaxPool
class GlobaLPool(nn.Module):
    """
    结合全局平均池化和全局最大池化的全局池化层。
    concat(GlobalAvgPool, GlobalMaxPool) -> Linear -> SiLU 激活
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.fused_proj = nn.Linear(d_model * 2, d_model)
        self.activation = nn.SiLU()
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fused_proj.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, d_model)
        avg_pool = torch.mean(x, dim=1)  # 全局平均池化 (捕捉平均特征)
        max_pool = torch.max(x, dim=1)[0] # 全局最大池化 (捕捉峰值特征)
        fused = torch.cat([avg_pool, max_pool], dim=1)
        output = self.activation(self.fused_proj(fused))
        return output

    
# Gated Attention Pooling
class GatedAttentionPool(nn.Module):
    """
    用 SwiGLU 替代原来简单的 Linear(d_model, 1) 来计算注意力分数
    """
    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = int(8 * d_model // 3)

        self.gate_value_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.to_score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.SiLU()

        nn.init.kaiming_normal_(self.gate_value_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.to_score.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: SwiGLU 非线性变换 → [B, L, hidden_dim]
        combined = self.gate_value_proj(x)                    # [B, L, 2*hidden]
        gate, value = combined.chunk(2, dim=-1)               # [B, L, hidden]
        hidden = self.activation(gate) * value                # SwiGLU
        # Step 2: 映射到注意力分数 logit
        scores = self.to_score(hidden).squeeze(-1)             # [B, L]
        # Step 3: softmax 得到权重
        weights = F.softmax(scores, dim=1).unsqueeze(1)        # [B, 1, L]
        # Step 4: 加权求和
        output = torch.bmm(weights, x).squeeze(1)              # [B, D]
        output = self.dropout(output)
        return output

# Gated Attention Pooling V2
class GatedAttentionPoolV2(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = int(8 * d_model // 3)

        self.proj = nn.Linear(d_model, 3 * hidden_dim, bias=False)  # gate, value_for_glu, value_for_pool
        self.to_score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.act = nn.SiLU()

        nn.init.kaiming_normal_(self.proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.to_score.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, L, 3*hidden]
        gate, glu_val, pool_val = self.proj(x).chunk(3, dim=-1)
        
        hidden = self.act(gate) * glu_val
        scores = self.to_score(hidden).squeeze(-1)
        attn = F.softmax(scores, dim=1).unsqueeze(1)  # [B,1,L]
        
        out = torch.bmm(attn, pool_val).squeeze(1)
        return self.dropout(out)


# 简单 MLP 分类头
class Simple_MLP_Classifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 2):
        super().__init__()
        # self.classifier = nn.Sequential(
        #       RMSNorm(d_model),  # 稳定训练
        #       nn.Linear(d_model, 4 * d_model),
        #       nn.SiLU(),
        #       nn.Dropout(p=0.1),
        #       nn.Linear(4 * d_model, num_classes)
        # )
        self.classifier = nn.Linear(d_model, num_classes) # 直接线性分类，避免过拟合

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        return logits

class SwiGLU_Classifier(nn.Module):
    """
    输入: (B, d_model)
    输出: (B, num_classes), 通过 SwiGLU Adapter + RMSNorm + Dropout + Linear 分类头
    """
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        expand_ratio: float,
        dropout: float,
    ):
        super().__init__()
        adapter_dim = int(d_model * expand_ratio)
        
        self.adapter = nn.Linear(d_model, 2 * adapter_dim, bias=True)
        self.proj_down = nn.Linear(adapter_dim, d_model, bias=True)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.proj_down.weight, mode='fan_in', nonlinearity='linear')
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = self.adapter(x)              # (B, 2*hidden)
        gate, value = combined.chunk(2, dim=-1) # (B, hidden)
        y = F.silu(gate) * value                # (B, hidden)
        
        out = self.proj_down(y) + x             # residual
        out = self.norm(out)
        out = self.dropout(out)
        
        return self.classifier(out)
