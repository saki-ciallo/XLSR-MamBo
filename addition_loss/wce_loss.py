import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失 (Weighted Cross Entropy Loss)
    
    用于处理类别不平衡问题，给少数类更高的权重
    
    参数:
        weight: 各类别权重，形状为 [num_classes]
                例如 [0.1, 0.9] 表示 class 0 权重 0.1，class 1 权重 0.9
        num_classes: 类别数量
        auto_weight: 是否根据数据集自动计算权重
    """
    def __init__(self, weight=None, num_classes=2, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        if weight is not None:
            self.register_buffer('weight', torch.FloatTensor(weight))
        else:
            self.weight = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出，形状 [B, num_classes]
            targets: 真实标签，形状 [B]
        Returns:
            loss: 加权交叉熵损失
        """
        return F.cross_entropy(logits, targets, weight=self.weight, reduction=self.reduction)

