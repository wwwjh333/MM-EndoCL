import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='laplacian', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        kernel_type: 'gaussian' | 'laplacian' | 'linear'
        kernel_num:  用于多核 Gaussian / Laplacian，linear 时会被忽略
        """
        super(MMD_loss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    # 原来的 gaussian，多核 RBF
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)  # [N, D]

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)  # [N, N]

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            # 避免除零
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + 1e-8)

        bandwidth /= (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)


    def laplacian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)  # [N, D]

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L1_distance = (total0 - total1).abs().sum(2)  # [N, N]

        if fix_sigma is not None:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L1_distance.data) / (n_samples ** 2 - n_samples + 1e-8)

        bandwidth /= (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L1_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    # 新增：Linear kernel，对应只对齐一阶矩的非线性弱一点的 MMD
    def linear_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)  # [N, D]
        K = total @ total.t()  # [N, N]
        return K

    def forward(self, source, target):
        batch_size = int(source.size(0))

        # 根据 kernel_type 选择核函数
        if self.kernel_type == 'gaussian':
            kernels = self.gaussian_kernel(
                source, target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma
            )
        elif self.kernel_type == 'laplacian':
            kernels = self.laplacian_kernel(
                source, target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma
            )
        elif self.kernel_type == 'linear':
            kernels = self.linear_kernel(source, target)
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
