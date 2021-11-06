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
    def gen_labels_cam(self, features, num_camids):
        # features 12936*2048  camids 12936
        feats_list = torch.split(features, num_camids, dim=0)
        labels_cam = []
        for feats in feats_list:
            labels_camid = self.gen_labels(feats)[0]
            labels_cam.append(labels_camid)
        labels_cam = torch.cat([labels for labels in labels_cam])
        return labels_cam
