import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from ..utils import MemoryLayer
from reid.utils import concat_all_gather
from mmcv.runner import get_dist_info


@HEADS.register_module()
class HybridHead(nn.Module):

    def __init__(self,
                 temperature_spcl=0.05,
                 temperature_tmp=0.1,
                 momentum=0.2,
                 feat_dim=2048,
                 memory_size=65536):
        super(HybridHead, self).__init__()
        self.temperature_spcl = temperature_spcl
        self.temperature_tmp = temperature_tmp
        self.momentum = momentum
        self.feat_dim = feat_dim
        self.memory_size = memory_size

        self.register_buffer(
            'features', torch.zeros((memory_size, feat_dim),
                                    dtype=torch.float))
        self.register_buffer('labels',
                             torch.zeros(memory_size, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, **kwargs):
        pass

    @torch.no_grad()
    def update_features(self, features):
        features = F.normalize(features, dim=1)
        self.features.data.copy_(features.data)

    @torch.no_grad()
    def update_labels(self, labels):
        self.labels.data.copy_(labels.data)

    def forward(self, inputs, feats_aug, idx, label, **kwargs):
        inputs = F.normalize(inputs, dim=1)

        # inputs: B*2048, features: L*2048
        inputs = MemoryLayer.apply(inputs, idx, self.features, self.momentum)
        inputs /= self.temperature_spcl
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums

        targets = self.labels[idx].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels,
                        torch.ones(self.memory_size, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(),
                                    mask.t().contiguous())
        loss_spcl = F.nll_loss(torch.log(masked_sim + 1e-6), targets)
        loss_contrastive = self.contrastiveloss(feats_aug, label)

        loss = loss_spcl + loss_contrastive


        return dict(loss=loss)

    def contrastiveloss(self, features, label, **kwargs):
        N = features.shape[0]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        label = concat_all_gather(label)
        label = label.view(-1, 1)

        label_mask = label.eq(label.t()).float()
        label_mask = label_mask.repeat(2, 2)
        is_neg = 1 - label_mask
        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask.bool(),
                                       mask == 1).reshape(2 * N, -1)
        neg_mask = torch.masked_select(is_neg.bool(),
                                       mask == 1).reshape(2 * N, -1)

        rank, world_size = get_dist_info()
        size = int(2 * N / world_size)

        pos_mask = torch.split(pos_mask, [size] * world_size, dim=0)[rank]
        neg_mask = torch.split(neg_mask, [size] * world_size, dim=0)[rank]
        logit = torch.split(logit, [size] * world_size, dim=0)[rank]

        n = logit.size(0)
        loss = []

        for i in range(n):
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                negative = logit[i, neg_inds].unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= self.temperature_tmp
                _label = _logit.new_zeros((1, ), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss = sum(loss)
        loss /= logit.size(0)

        return loss
