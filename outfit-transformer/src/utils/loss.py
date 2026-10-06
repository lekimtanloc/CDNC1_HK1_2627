import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma

    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, self.gamma, reduction='mean')

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight), self.gamma)



class SetWiseOutfitLoss(nn.Module):
    def __init__(self, margin=2):
        super().__init__()
        self.margin = margin
        self.pdist = nn.PairwiseDistance(p=2)


    def forward(self, target_embedding:torch.Tensor, pos_embedding:torch.Tensor, neg_embeddings:torch.Tensor, device):

        nb_neg_samples = neg_embeddings.shape[0]

        # compute the distance of the positive embedding to the target embedding 
        dist_t_fp = self.pdist(target_embedding, pos_embedding)
        
        # repeat the target embedding to match the size of negative samples embeddings
        t = target_embedding.repeat(nb_neg_samples,1)
        # print('t shape after repeat : {}'.format(t.shape))

        # compute the distance of the negative samples embeddings to the target embedding
        dist_t_fnn = self.pdist(t, neg_embeddings)
        # print('shape dist_t_fnn : {}'.format(dist_t_fnn.shape))

        pos_neg_dist = torch.sub(dist_t_fp, dist_t_fnn)
        pos_neg_dist = torch.add(pos_neg_dist, self.margin)

        loss = torch.sum(torch.maximum(torch.zeros(nb_neg_samples).to(device), pos_neg_dist)) / nb_neg_samples

        return loss



class SetWiseOutfitLossHard(nn.Module):
    def __init__(self, margin=2):
        super().__init__()
        self.margin = margin
        self.pdist = nn.PairwiseDistance(p=2)


    def forward(self, target_embedding:torch.Tensor, pos_embedding:torch.Tensor, neg_embeddings:torch.Tensor, device):

        nb_neg_samples = neg_embeddings.shape[0]

        # compute the distance of the positive embedding to the target embedding 
        dist_t_fp = self.pdist(target_embedding, pos_embedding)
        
        # repeat the target embedding to match the size of negative samples embeddings
        t = target_embedding.repeat(nb_neg_samples,1)
        # print('t shape after repeat : {}'.format(t.shape))

        # compute the distance of the negative samples embeddings to the target embedding
        dist_t_fnn = self.pdist(t, neg_embeddings)
        # print('shape dist_t_fnn : {}'.format(dist_t_fnn.shape))

        # compute the min of distances of the negative samples embeddings to the target embedding
        min_dist_t_fnn = torch.min(dist_t_fnn)

        loss_hard = torch.add(torch.sub(dist_t_fp, min_dist_t_fnn), self.margin)

        pos_neg_dist = torch.sub(dist_t_fp, dist_t_fnn)
        pos_neg_dist = torch.add(pos_neg_dist, self.margin)

        loss_all = torch.sum(torch.maximum(torch.zeros(nb_neg_samples).to(device), pos_neg_dist)) / nb_neg_samples

        loss = loss_all + loss_hard

        return loss







                
