from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from ..distances import jaccard_distance
from .builder import LABEL_GENERATORS
from .self_paced import SelfPacedGenerator


@LABEL_GENERATORS.register_module()
class CamAwareSelfPacedGenerator(SelfPacedGenerator):

    @torch.no_grad()
    def dbscan_single(self, features, dist, eps, cam_aware=False):
        assert isinstance(dist, np.ndarray)

        if cam_aware:
            self.use_outliers = False

        cluster = DBSCAN(
            eps=eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=-1)
        labels = cluster.fit_predict(dist)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # cluster labels -> pseudo labels
        # compute cluster centers
        centers = defaultdict(list)
        outliers = 0
        for i, label in enumerate(labels):
            if label == -1:
                if not self.use_outliers:
                    continue
                labels[i] = num_clusters + outliers
                outliers += 1

            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0)
            for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)
        labels = torch.from_numpy(labels).long()
        num_clusters += outliers

        self.use_outliers = True

        return labels, centers, num_clusters

    @torch.no_grad()
    def gen_labels_cam(self, features, num_camids):
        # features 12936*2048  camids 12936
        feats_list = torch.split(features, num_camids, dim=0)
        labels_cam = []
        for feats in feats_list:
            labels_camid = self.gen_labels(feats, cam_aware=True)[0]
            labels_cam.append(labels_camid)
        labels_cam = torch.cat([labels for labels in labels_cam])
        return labels_cam

    @torch.no_grad()
    def gen_labels(self, features, cam_aware=False):
        dist = jaccard_distance(features, k1=self.k1, k2=self.k2)

        if len(self.eps) == 1:
            # normal clustering
            labels, centers, num_classes = self.dbscan_single(
                features, dist, self.eps[0], cam_aware)
            return labels, centers, num_classes, None
        elif len(self.eps) == 3:
            (labels_normal, centers, num_classes,
             indep_thres) = self.dbscan_self_paced(features, dist, self.eps)
            return labels_normal, centers, num_classes, indep_thres
