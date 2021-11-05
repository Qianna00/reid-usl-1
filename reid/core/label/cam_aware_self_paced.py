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
    def gen_labels_cam(self, features, camids):
        # features 12936*2048  camids 12936
        unqiue_camids = list(set(camids))
        labels_cam = []
        for camid in unqiue_camids:
            camid_index = torch.nonzero(torch.tensor(camids) - camid).squeeze()
            feat_camid = features[camid_index]
            print(features[camid_index].size())
            labels_camid = self.gen_labels(feat_camid)[0]
            labels_cam.extend(labels_camid)
        return labels_cam
