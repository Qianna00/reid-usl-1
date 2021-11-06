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
            camid_index = (camids == camid).nonzero(as_tuple=True)[0]
            feat_camid = features[camid_index]
            labels_camid = self.gen_labels(feat_camid)[0]
            print(type(labels_camid))
            labels_cam.append(labels_camid)
        print(type(labels_cam), type(labels_cam[0], labels_cam[0].size()))
        return labels_cam
