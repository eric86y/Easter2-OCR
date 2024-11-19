import torch
from torch import nn

class CustomCTC(nn.Module):
    def __init__(self, gamma: float = 0.5, alpha: float = 0.25, blank: int = 0, reduction: str = "sum", zero_infinity: bool = True):
        super(CustomCTC, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, labels, input_lengths, target_lengths):
        ctc_loss = nn.CTCLoss(blank=self.blank, reduction=self.reduction, zero_infinity=self.zero_infinity)(
            log_probs, labels, input_lengths, target_lengths
        )
        p = torch.exp(-ctc_loss)
        loss = self.alpha * (torch.pow((1 - p), self.gamma)) * ctc_loss

        return loss