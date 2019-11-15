import torch

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                       - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)) + eps)

def distance_vectors_pairwise(anchor, positive, negative=None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)
    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2 * torch.sum(anchor * positive, dim=1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2 * torch.sum(anchor * negative, dim=1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2 * torch.sum(positive * negative, dim=1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p

def loss_HardNet(anchor, positive, margin=1.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) + eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10

    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    dist_without_min_on_diag = dist_without_min_on_diag + mask

    min_neg = torch.min(dist_without_min_on_diag, 1)[0]
    min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
    min_neg = torch.min(min_neg, min_neg2)

    loss = torch.clamp(margin + pos - min_neg, min=0.0)
    loss = torch.mean(loss)

    return loss
