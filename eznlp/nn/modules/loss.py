# -*- coding: utf-8 -*-
import torch

from ..functional import (
    focal_loss,
    smooth_label_cross_entropy,
    soft_label_cross_entropy,
)


class SoftLabelCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight: torch.Tensor = None, reduction: str = "none"):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)

    def extra_repr(self):
        return f"weight={self.weight}"

    def forward(self, logits: torch.Tensor, soft_target: torch.Tensor):
        return soft_label_cross_entropy(
            logits, soft_target, weight=self.weight, reduction=self.reduction
        )


class SmoothLabelCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        epsilon: float = 0.1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "none",
    ):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def extra_repr(self):
        return f"epsilon={self.epsilon}, weight={self.weight}"

    def forward(self, logits: torch.Tensor, target: torch.LongTensor):
        return smooth_label_cross_entropy(
            logits,
            target,
            epsilon=self.epsilon,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        gamma: float = 0.0,
        weight: float = None,
        ignore_index: int = -100,
        reduction: str = "none",
    ):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def extra_repr(self):
        return f"gamma={self.gamma}, weight={self.weight}"

    def forward(self, logits: torch.Tensor, target: torch.LongTensor):
        return focal_loss(
            logits,
            target,
            gamma=self.gamma,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class SoftLabelFocalLoss(torch.nn.modules.loss._WeightedLoss):
    """Focal Loss with soft label support for boundary selection.
    
    Combines soft label cross entropy with focal loss weighting to address
    class imbalance in boundary selection tasks.
    
    Args:
        gamma: Focusing parameter for modulating factor (1-p_t)^gamma. 
               gamma=0 is equivalent to soft label cross entropy.
        weight: A manual rescaling weight given to each class.
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum'.
    """
    
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "none"):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)
        self.gamma = gamma
    
    def extra_repr(self):
        return f"gamma={self.gamma}, weight={self.weight}"
    
    def forward(self, logits: torch.Tensor, soft_target: torch.Tensor):
        """
        Args:
            logits: (*, num_classes) unnormalized logits
            soft_target: (*, num_classes) soft label distribution (sums to 1 along last dim)
        
        Returns:
            Focal loss weighted by (1 - p_t)^gamma
        """
        # 计算 log softmax
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # 计算标准交叉熵: -sum(soft_target * log_probs)
        ce_loss = -(soft_target * log_probs).sum(dim=-1)
        
        # 计算聚焦权重 (1 - p_t)^gamma
        # p_t 是 soft target 加权的预测概率
        p_t = (soft_target * probs).sum(dim=-1)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.weight is not None:
            # weight 应用于每个样本（如果需要类别权重，可以扩展）
            loss = loss * self.weight
        
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
