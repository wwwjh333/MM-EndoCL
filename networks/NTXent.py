import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch import nn

# class NTXent(Module):
#     # adapted from https://github.com/clabrugere/pytorch-scarf/blob/master/scarf/loss.py but rewrite the loss to avoid
#     # explicit log(exp(.)) and log(sum(exp(.))) operations to improve numerical stability.
#     def __init__(self, temperature=1.0):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, z_1, z_2,z_3,z_4):
#         batch_size, device = z_1.size(0), z_1.device
#
#         # compute similarities between all the 2N views
#         z = torch.cat([z_1, z_2], dim=0)  # (2 * bs, dim_emb)
#         similarity = F.cosine_similarity(z[:, None], z[None, :], dim=2) / self.temperature  # (2 * bs, 2 * bs)
#         sim_ij = torch.diag(similarity, batch_size)  # (bs,)
#         sim_ji = torch.diag(similarity, -batch_size)  # (bs,)
#
#         z1 = torch.cat([z_3, z_4], dim=0)  # (2 * bs, dim_emb)
#         similarity_1 = F.cosine_similarity(z1[:, None], z1[None, :], dim=2) / self.temperature  # (2 * bs, 2 * bs)
#         sim_ij_1 = torch.diag(similarity_1, batch_size)  # (bs,)
#         sim_ji_1 = torch.diag(similarity_1, -batch_size)  # (bs,)
#
#         # positive contains the 2N similarities between two views of the same sample
#         positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2 * bs,)
#
#         negatives_1 = torch.cat([sim_ij_1, sim_ji_1], dim=0)
#
#         # negative contains the (2N, 2N - 1) similarities between the view of a sample and all the other views that are
#         # not from that same sample
#         mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)  # (2 * bs, 2 * bs)
#         negatives = similarity[mask].view(2 * batch_size, 2 * batch_size - 1)  # (2 * bs, 2 * bs - 1)
#
#         # the loss can be rewritten as the sum of the alignement loss making the two representations of the same
#         # sample closer, and the distribution loss making the representations of different samples farther
#         loss_alignement = -torch.mean(positives)
#         loss_distribution = torch.mean(torch.logsumexp(negatives, dim=1))
#         loss_negatives = torch.mean(negatives_1)
#         loss = 1*loss_alignement + 0.5*loss_distribution+1*loss_negatives
#
#         return loss,loss_alignement,loss_distribution,loss_negatives

# class NTXent(nn.Module):
#
#     def __init__(self, temperature: float = 0.1):
#         super().__init__()
#
#         self.temperature = temperature
#
#     def mask(self, n_batch, device):
#         mask_self = torch.eye(2 * n_batch, dtype=torch.bool, device=device)
#         mask_pos = mask_self.roll(shifts=n_batch, dims=1)
#         mask_neg = mask_self + mask_pos
#         return mask_pos, mask_self, ~mask_neg
#
#     def forward(self, p1, p2,p3,p4):
#         n_batch, _ = p1.shape
#
#         p1, p2,p3,p4 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1),F.normalize(p3, dim=-1), F.normalize(p4, dim=-1)
#
#         z = torch.cat([p1, p2], dim=0)
#         z1 = torch.cat([p3, p4], dim=0)
#         sim_mat = torch.matmul(z, z.transpose(-2, -1)) / self.temperature
#         sim_mat1 = torch.matmul(z1, z1.transpose(-2, -1)) / self.temperature
#
#         mask_pos, mask_self, mask_neg = self.mask(n_batch, device=p1.device)
#         sim_mat[mask_self] = float('-inf')
#         sim_mat1[mask_self] = float('-inf')
#
#         # loss = torch.logsumexp(sim_mat, dim=-1) - sim_mat[mask_pos]
#         loss = 0.05*sim_mat1[mask_pos]+torch.logsumexp(sim_mat, dim=-1) - sim_mat[mask_pos]
#         return loss.mean()

class NTXent(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()

        self.temperature = temperature

    def mask(self, n_batch, device):
        mask_self = torch.eye(4 * n_batch, dtype=torch.bool, device=device)
        mask_pos = mask_self.clone()
        for i in range(n_batch):
            mask_pos[i] = mask_pos[i].roll(shifts=n_batch, dims=0)
        for i in range(n_batch):
            mask_pos[n_batch+i] = mask_pos[n_batch+i].roll(shifts=3*n_batch, dims=0)
        mask_pos[2*n_batch:] = False
        mask_neg = mask_self + mask_pos
        return mask_pos, mask_self, ~mask_neg

    def forward(self, p1, p2,p3,p4):
        n_batch, _ = p1.shape

        p1, p2,p3,p4 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1),F.normalize(p3, dim=-1), F.normalize(p4, dim=-1)

        z = torch.cat([p1, p2,p3,p4], dim=0)
        sim_mat = torch.matmul(z, z.transpose(-2, -1)) / self.temperature

        mask_pos, mask_self, mask_neg = self.mask(n_batch, device=p1.device)
        sim_mat[mask_self] = float('-inf')

        # loss = torch.mean(torch.logsumexp(sim_mat, dim=-1))  - torch.mean(sim_mat[mask_pos])
        # 正样本损失：平均正样本相似度
        pos_loss = sim_mat[mask_pos].mean()

        # 负样本损失：使用 logsumexp 提高数值稳定性
        neg_loss = torch.logsumexp(sim_mat.masked_fill(~mask_neg, float('-inf')), dim=-1).mean()

        # 总损失 = 负样本损失 - 正样本损失
        loss = neg_loss - pos_loss
        return loss.mean()
    
# class NTXent(nn.Module):

#     def __init__(self, temperature: float = 0.1):
#         super().__init__()

#         self.temperature = temperature

#     def mask(self, n_batch, device):
#         mask_self = torch.eye(2 * n_batch, dtype=torch.bool, device=device)
#         mask_pos = mask_self.roll(shifts=n_batch, dims=1)
#         mask_neg = mask_self + mask_pos
#         return mask_pos, mask_self, ~mask_neg

#     def forward(self, p1, p2):
#         n_batch, _ = p1.shape

#         # 归一化处理
#         p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

#         # 拼接并计算相似度矩阵
#         z = torch.cat([p1, p2], dim=0)
#         sim_mat = torch.matmul(z, z.T) / self.temperature

#         # 获取掩码
#         mask_pos, mask_self, mask_neg = self.mask(n_batch, device=p1.device)

#         # 忽略自身的相似度
#         sim_mat.masked_fill_(mask_self, float('-inf'))

#         # 正样本损失：平均正样本相似度
#         pos_loss = sim_mat[mask_pos].mean()

#         # 负样本损失：使用 logsumexp 提高数值稳定性
#         neg_loss = torch.logsumexp(sim_mat.masked_fill(~mask_neg, float('-inf')), dim=-1).mean()

#         # 总损失 = 负样本损失 - 正样本损失
#         loss = neg_loss - pos_loss
#         return loss
    
# class NTXent(nn.Module):

#     def __init__(self, temperature: float = 0.1):
#         super().__init__()

#         self.temperature = temperature

#     def mask(self, n_batch, device):
#         mask_self = torch.eye(2 * n_batch, dtype=torch.bool, device=device)
#         mask_pos = mask_self.roll(shifts=n_batch, dims=1)
#         mask_neg = mask_self + mask_pos
#         return mask_pos, mask_self, ~mask_neg

#     def forward(self, p1, p2):
#         n_batch, _ = p1.shape

#         p1, p2 = F.normalize(p1, dim=-1), F.normalize(p2, dim=-1)

#         z = torch.cat([p1, p2], dim=0)
#         sim_mat = torch.matmul(z, z.transpose(-2, -1)) / self.temperature

#         mask_pos, mask_self, mask_neg = self.mask(n_batch, device=p1.device)
#         sim_mat[mask_self] = float('-inf')

#         loss = torch.logsumexp(sim_mat, dim=-1) - sim_mat[mask_pos]
#         return loss.mean()