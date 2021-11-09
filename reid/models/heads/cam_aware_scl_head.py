import torch
import torch.nn as nn
from mmcv.runner import get_dist_info

from reid.utils import concat_all_gather, concat_all_gather_async
from ..builder import HEADS
from .scl_head import AnotherSCLHead


@HEADS.register_module()
class CamAwareSCLHead(AnotherSCLHead):

    def forward(self, features, label, camid, **kwargs):
        N = features.shape[0]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        label = concat_all_gather(label)
        label = label.view(-1, 1)
        camid = concat_all_gather(camid)
        camid = camid.view(-1, 1)

        label_mask = label.eq(label.t())
        is_pos = label_mask & camid.eq(camid.t())
        label_mask = label_mask.float().repeat(2, 2)
        is_pos = is_pos.float().repeat(2, 2)
        is_neg = 1 - label_mask
        is_neg_cam = 1 - is_pos

        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask.bool(),
                                       mask == 1).reshape(2 * N, -1)
        neg_mask = torch.masked_select(is_neg.bool(),
                                       mask == 1).reshape(2 * N, -1)
        pos_mask_cam = torch.masked_select(is_pos.bool(),
                                       mask == 1).reshape(2 * N, -1)
        neg_mask_cam = torch.masked_select(is_neg_cam.bool(),
                                       mask == 1).reshape(2 * N, -1)

        rank, world_size = get_dist_info()
        size = int(2 * N / world_size)

        pos_mask = torch.split(pos_mask, [size] * world_size, dim=0)[rank]
        neg_mask = torch.split(neg_mask, [size] * world_size, dim=0)[rank]
        logit = torch.split(logit, [size] * world_size, dim=0)[rank]
        pos_mask_cam = torch.split(pos_mask_cam, [size] * world_size, dim=0)[rank]
        neg_mask_cam = torch.split(neg_mask_cam, [size] * world_size, dim=0)[rank]

        n = logit.size(0)
        loss = []

        for i in range(n):
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)
            pos_inds_cam = torch.nonzero(pos_mask_cam[i] == 1, as_tuple=False).view(-1)
            neg_inds_cam = torch.nonzero(neg_mask_cam[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                negative = logit[i, neg_inds].unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= self.temperature
                _label = _logit.new_zeros((1, ), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))
            loss_single_img_cam = []
            for k in range(pos_inds_cam.size(0)):
                positive_cam = logit[i, pos_inds_cam[k]].reshape(1, 1)
                negative_cam = logit[i, neg_inds_cam].unsqueeze(0)
                _logit_cam = torch.cat((positive_cam, negative_cam), dim=1)
                _logit_cam /= self.temperature
                _label_cam = _logit_cam.new_zeros((1,), dtype=torch.long)
                _loss_cam = self.criterion(_logit_cam, _label_cam)
                loss_single_img_cam.append(_loss_cam)
            loss.append(sum(loss_single_img_cam) / pos_inds_cam.size(0) * 0.5)

        loss = sum(loss)
        loss /= logit.size(0)

        return dict(loss=loss)


@HEADS.register_module()
class AnotherCamAwareSCLHead(CamAwareSCLHead):
    def forward(self, features, label, camid, **kwargs):
        N = features.shape[0]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        label = concat_all_gather(label)
        label = label.view(-1, 1)
        camid = concat_all_gather(camid)
        camid = camid.view(-1, 1)

        label_mask = label.eq(label.t())
        is_pos = label_mask & camid.eq(camid.t())
        label_mask = label_mask.float().repeat(2, 2)
        is_pos = is_pos.float().repeat(2, 2)
        is_neg = 1 - label_mask
        is_neg_cam = 1 - is_pos

        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask.bool(),
                                       mask == 1).reshape(2 * N, -1)
        neg_mask = torch.masked_select(is_neg.bool(),
                                       mask == 1).reshape(2 * N, -1)
        pos_mask_cam = torch.masked_select(is_pos.bool(),
                                       mask == 1).reshape(2 * N, -1)
        neg_mask_cam = torch.masked_select(is_neg_cam.bool(),
                                       mask == 1).reshape(2 * N, -1)

        rank, world_size = get_dist_info()
        size = int(2 * N / world_size)

        pos_mask = torch.split(pos_mask, [size] * world_size, dim=0)[rank]
        neg_mask = torch.split(neg_mask, [size] * world_size, dim=0)[rank]
        logit = torch.split(logit, [size] * world_size, dim=0)[rank]
        pos_mask_cam = torch.split(pos_mask_cam, [size] * world_size, dim=0)[rank]
        neg_mask_cam = torch.split(neg_mask_cam, [size] * world_size, dim=0)[rank]

        n = logit.size(0)
        loss = []

        for i in range(n):
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)
            pos_inds_cam = torch.nonzero(pos_mask_cam[i] == 1, as_tuple=False).view(-1)
            neg_inds_cam = torch.nonzero(neg_mask_cam[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                negative = logit[i, neg_inds].unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= self.temperature
                _label = _logit.new_zeros((1, ), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))
            loss_single_img_cam = []
            for k in range(pos_inds_cam.size(0)):
                positive_cam = logit[i, pos_inds_cam[k]].reshape(1, 1)
                negative_cam = logit[i, neg_inds_cam].unsqueeze(0)
                _logit_cam = torch.cat((positive_cam, negative_cam), dim=1)
                _logit_cam /= self.temperature
                _label_cam = _logit_cam.new_zeros((1,), dtype=torch.long)
                _loss_cam = self.criterion(_logit_cam, _label_cam)
                loss_single_img_cam.append(_loss_cam)
            loss.append(sum(loss_single_img_cam) / pos_inds_cam.size(0) * 0.2)

        loss = sum(loss)
        loss /= logit.size(0)

        return dict(loss=loss)
    

@HEADS.register_module()
class AnotherNewCamAwareSCLHead(AnotherSCLHead):
    def forward(self, features, label, camid, label_cam, **kwargs):
        loss_global = self.compute_loss(features, label)
        loss_intra_cam = self.compute_intra_loss(features, label_cam, camid)
        loss_inter_cam = self.compute_inter_loss(features, label_cam, camid)
        return dict(loss_intra_cam=loss_intra_cam, loss_inter_cam=loss_inter_cam, loss_global=loss_global)

    def compute_loss(self, features, label, **kwargs):
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
                _logit /= self.temperature
                _label = _logit.new_zeros((1,), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss = sum(loss)
        loss /= logit.size(0)

        return loss

    def compute_intra_loss(self, features, label, camid):
        N = features.shape[0]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)
        label = concat_all_gather(label)
        # label = label[torch.nonzero(label != -1)]
        label = label.view(-1, 1)

        global_label_mask = label.eq(label.t())
        camid = concat_all_gather(camid)
        camid = camid.view(-1, 1)
        cam_mask = camid.eq(camid.t())
        # mask unclusters
        label_uncluster = -torch.ones(label.size()).cuda()
        uncluster_mask = (1 - label.eq(label_uncluster.t()).float()).bool()
        cluster_label_mask = global_label_mask & cam_mask & uncluster_mask
        cluster_is_neg = (1 - global_label_mask.float()).bool() & cam_mask & uncluster_mask
        label_mask = cluster_label_mask.repeat(2, 2)
        is_neg = cluster_is_neg.repeat(2, 2)

        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask,
                                       mask == 1).reshape(2 * N, -1)
        neg_mask = torch.masked_select(is_neg,
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
                _logit /= self.temperature
                _label = _logit.new_zeros((1,), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss = sum(loss)
        loss /= logit.size(0)
        return loss

    def compute_inter_loss(self, features, label, camid):
        N= features.shape[0]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        label = concat_all_gather(label)
        label = label.view(-1, 1)
        camid = concat_all_gather(camid)
        camid = camid.view(-1, 1)
        cam_mask = camid.eq(camid.t())
        label_uncluster = -torch.ones(label.size()).cuda()
        uncluster_mask = (1 - label.eq(label_uncluster.t()).float()).bool()
        diff_cam_mask = (1 - cam_mask.float()).bool()

        label_mask = label.eq(label.t()) & uncluster_mask
        is_neg = (1 - label_mask.float()).bool() & diff_cam_mask & uncluster_mask
        label_mask = label_mask.repeat(2, 2)
        is_neg = is_neg.repeat(2, 2)
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
                negative, _ = torch.topk(negative, k=100, dim=1)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= self.temperature
                _label = _logit.new_zeros((1,), dtype=torch.long)
                _loss = self.criterion(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss = sum(loss)
        loss /= logit.size(0)

        return loss
