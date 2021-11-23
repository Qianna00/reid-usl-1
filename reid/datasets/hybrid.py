import copy

import torch

from .builder import DATASETS
from .pseudo_label import PseudoLabelDataset
from .pipelines import build_pipeline


@DATASETS.register_module()
class HybridDataset(PseudoLabelDataset):
    def __init__(self, data_source, pipeline=None, aug_pipeline=None, test_mode=False):

        super(HybridDataset, self).__init__(data_source=data_source, pipeline=pipeline, test_mode=test_mode)

        if aug_pipeline is not None:
            self.aug_pipeline = build_pipeline(aug_pipeline, dataset=self)
        else:
            self.aug_pipeline = None


    def __getitem__(self, idx):
        img, pid, camid = self.get_sample(idx)
        label = self.pid_dict[pid] if not self.test_mode else pid
        results = dict(img=img, label=label, pid=pid, camid=camid, idx=idx)

        img1 = self.aug_pipeline(copy.deepcopy(results))['img']
        img2 = self.aug_pipeline(copy.deepcopy(results))['img']
        img_cat = torch.cat((img.unsqueeze(0), img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        results['img'] = img_cat

        return results