import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math


class ClfDistillLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, hidden, logits, bias, teach_probs, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()


class Plain(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias, teacher_probs, labels, reduce=True):
        return F.cross_entropy(logits, labels, reduce=reduce)