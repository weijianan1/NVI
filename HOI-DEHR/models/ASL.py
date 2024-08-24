import torch
import torch.nn as nn

# class AsymmetricLossOptimized(nn.Module):
#     ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
#     favors inplace operations'''

#     def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, alpha=0.25, eps=1e-8, disable_torch_grad_focal_loss=False):
#         super(AsymmetricLossOptimized, self).__init__()

#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.alpha = alpha
#         self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
#         self.eps = eps

#         # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
#         self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

#     def forward(self, x, y):
#         """"
#         Parameters
#         ----------
#         x: input logits
#         y: targets (multi-label binarized vector)
#         """

#         self.targets = y
#         self.anti_targets = 1 - y

#         # Calculating Probabilities
#         # self.xs_pos = torch.sigmoid(x)
#         self.xs_pos = x
#         self.xs_neg = 1.0 - self.xs_pos

#         # Asymmetric Clipping
#         if self.clip is not None and self.clip > 0:
#             self.xs_neg.add_(self.clip).clamp_(max=1)

#         # Basic CE calculation
#         self.loss = self.alpha * self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
#         self.loss.add_((1 - self.alpha) * self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

#         # 因为用了targets作为mask所以可以直接加起来
#         # Asymmetric Focusing
#         if self.gamma_neg > 0 or self.gamma_pos > 0:
#             if self.disable_torch_grad_focal_loss:
#                 torch.set_grad_enabled(False)
#             self.xs_pos = self.xs_pos * self.targets
#             self.xs_neg = self.xs_neg * self.anti_targets
#             self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
#                                           self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
#             if self.disable_torch_grad_focal_loss:
#                 torch.set_grad_enabled(True)
#             self.loss *= self.asymmetric_w

#         num_pos = y.eq(1).float().sum()

#         if num_pos == 0:
#             loss = -(self.loss * (1 - y)).sum()
#         else:
#             loss = -self.loss.sum() / num_pos
#         return loss


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, alpha=0.25, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha = alpha
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        # self.xs_pos = torch.sigmoid(x)
        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            # self.xs_neg.add_(self.clip).clamp_(max=1)
            self.xs_neg.add_(self.clip).clamp_(min=1e-4, max=1-1e-4)

        # Basic CE calculation
        self.loss = self.alpha * self.targets * torch.log(self.xs_pos)
        self.loss.add_((1 - self.alpha) * self.anti_targets * torch.log(self.xs_neg))

        # 因为用了targets作为mask所以可以直接加起来
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        num_pos = y.eq(1).float().sum()

        if num_pos == 0:
            loss = -(self.loss * (1 - y)).sum()
        else:
            loss = -self.loss.sum() / num_pos
        return loss

    # def forward(self, pred, gt, weights=None):
    #     ''' Modified focal loss. Exactly the same as CornerNet.
    #     Runs faster and costs a little bit more memory
    #     '''
    #     pos_inds = gt.eq(1).float()
    #     neg_inds = gt.lt(1).float()

    #     loss = 0

    #     pos_loss = self.alpha * torch.log(pred) * torch.pow(1 - pred, self.gamma_pos) * pos_inds
    #     if weights is not None:
    #         pos_loss = pos_loss * weights[:-1]

    #     # neg_loss = (1 - self.alpha) * torch.log(1 - pred) * torch.pow(pred, self.gamma_neg) * neg_inds
    #     # pred_m = torch.max((pred - 0.05), 0)[0]
    #     pred_m = torch.clamp((pred - 0.05), min=1e-4, max=1-1e-4)
    #     neg_loss = (1 - self.alpha) * torch.log(1 - pred_m) * torch.pow(pred_m, self.gamma_neg) * neg_inds

    #     num_pos = pos_inds.float().sum()
    #     pos_loss = pos_loss.sum()
    #     neg_loss = neg_loss.sum()

    #     if num_pos == 0:
    #         loss = loss - neg_loss
    #     else:
    #         loss = loss - (pos_loss + neg_loss) / num_pos
    #     return loss



