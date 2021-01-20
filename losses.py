import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        mask = target >= 0
        output1 = output1[mask,:]
        output2 = output2[mask,:]
        target = target[mask]
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class DiceLoss(nn.Module):
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        """ Compute the dice score of a segmentation result.
        Args:
            input (torch.FloatTensor): Segmentation result.
            target (torch.FloatTensor): Segmentation mask (ground truth).
        Returns:
            float: [description]
        """
        input = F.sigmoid(input)
        smooth = 1.
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class Weighted_BCE(nn.Module):
  def __init__(self):
    super(Weighted_BCE, self).__init__()
  
  def forward(self, input, target, pos_weight):
    bce_loss = nn.BCEWithLogitsLoss(input,target,pos_weight=pos_weight)
    return bce_loss


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        targets = targets.type_as(outputs)
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss






