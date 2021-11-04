import os.path as osp
import re
from glob import glob

from ..builder import DATA_SOURCES
from .reid_data_source import ReIDDataSource
from collections import OrderedDict


@DATA_SOURCES.register_module()
class Market1501(ReIDDataSource):
    """Market-1501.

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    DATA_SOURCE = 'Market1501'
    NUM_CAMERAS = 6

    def __init__(self, data_root, cam_aware=False):
        self.data_root = data_root

        self.train_dir = osp.join(self.data_root, 'bounding_box_train')
        self.query_dir = osp.join(self.data_root, 'query')
        self.gallery_dir = osp.join(self.data_root, 'bounding_box_test')
        self.cam_aware = cam_aware

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(Market1501, self).__init__(train, query, gallery)

    def parse_data_cam_aware(self, data):
        pids = set()
        camids = set()
        cam_ids = list()
        for i, (_, pid, camid, _) in enumerate(data):
            pids.add(pid)
            camids.add(camid)
            cam_ids.append(camid)

        return list(pids), list(camids), cam_ids

    def _get_train_data(self, verbose=True):
        if self.cam_aware:
            pids, camids, cam_ids = self.parse_data_cam_aware(self.train)
        else:
            pids, camids = self.parse_data(self.train)

        if verbose:
            self._print_train_info(len(self.train), len(pids), len(camids))
        if self.cam_aware:
            return self.train, pids, camids, cam_ids
        else:
            return self.train, pids, camids

    def process_dir(self, dir_path):
        img_paths = glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        if self.cam_aware:
            data_cam_aware = {'cam' + str(i): [] for i in range(self.NUM_CAMERAS)}
            data_cam_aware = OrderedDict(data_cam_aware)

            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1:
                    continue
                data_cam_aware['cam' + str(camid-1)].append((img_path, pid, camid, pid))
            for v in data_cam_aware.values():
                data.extend(v)
        else:
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1:
                    continue
                data.append((img_path, pid, camid))

        return data
