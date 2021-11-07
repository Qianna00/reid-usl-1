import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook

from ..label import build_label_generator
from .extractor import Extractor


@HOOKS.register_module()
class LabelGenerationHook(Hook):

    def __init__(self, extractor, label_generator, start=1, interval=1):
        self.extractor = Extractor(**extractor)
        self.label_generator = build_label_generator(label_generator)
        self.start = start
        self.interval = interval

        self.distributed = dist.is_available() and dist.is_initialized()
        self.cam_aware = self.extractor.dataset.data_source.cam_aware

    @torch.no_grad()
    def _dist_gen_labels(self, feats):
        if dist.get_rank() == 0:
            labels = self.label_generator.gen_labels(feats)[0]
            labels = labels.cuda()
        else:
            labels = torch.zeros(feats.shape[0], dtype=torch.long).cuda()
        dist.broadcast(labels, 0)

        return labels

    @torch.no_grad()
    def _dist_gen_labels_cam(self, feats, num_camids):
        if dist.get_rank() == 0:
            labels_cam = self.label_generator.gen_labels_cam(feats, num_camids)
            labels_cam = labels_cam.cuda()
        else:
            labels_cam = torch.zeros(feats.shape[0], dtype=torch.long).cuda()
        dist.broadcast(labels_cam, 0)

        return labels_cam

    @torch.no_grad()
    def _non_dist_gen_labels(self, feats):
        labels = self.label_generator.gen_labels(feats)[0]

        return labels.cuda()

    def update_flag(self, runner):
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                return False  # No evaluation
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False
        return True

    def before_train_epoch(self, runner):
        if not self.update_flag(runner):
            return

        with torch.no_grad():
            feats = self.extractor.extract_feats(runner.model)
            if self.cam_aware:
                num_camids = runner.data_loader.dataset.num_camids
            else:
                num_camids = None
            if self.distributed:
                labels = self._dist_gen_labels(feats)
                if self.cam_aware:
                    labels_cam = self._dist_gen_labels_cam(feats, num_camids)
            else:
                labels = self._non_dist_gen_labels(feats)

        runner.model.train()

        labels = labels.cpu().numpy()
        if self.cam_aware:
            labels_cam = labels_cam.cpu().numpy()
        else:
            labels_cam = None
        runner.data_loader.dataset.update_labels(labels, labels_cam)
        if hasattr(runner.data_loader.sampler, 'init_data'):
            # identity sampler cases
            runner.data_loader.sampler.init_data()

        self.evaluate(runner, labels, labels_cam, num_camids)

    def evaluate(self, runner, labels, labels_cam=None, num_camids=None):
        hist = np.bincount(labels)
        clusters = np.where(hist > 1)[0]
        unclusters = np.where(hist == 1)[0]
        runner.logger.info(f'{self.__class__.__name__}: '
                           f'{clusters.shape[0]} clusters, '
                           f'{unclusters.shape[0]} unclusters')
        if labels_cam is not None:
            camids_split_index = []
            num = 0
            for i in range(len(num_camids)-1):
                num += num_camids[i]
                camids_split_index.append(num)
            labels_cam_list = np.split(labels_cam, camids_split_index)
            for i, labels_camid in enumerate(labels_cam_list):
                hist_camid = np.bincount(labels_camid)
                clusters_camid = np.where(hist_camid > 1)[0]
                unclusters_camid = np.where(hist_camid == 1)[0]
                runner.logger.info(f'{self.__class__.__name__} cam{i}: '
                                   f'{clusters_camid.shape[0]} clusters, '
                                   f'{unclusters_camid.shape[0]} unclusters')

