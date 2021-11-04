from .builder import DATASETS
from .reid_dataset import ReIDDataset


@DATASETS.register_module()
class PseudoLabelDataset(ReIDDataset):

    def update_labels(self, labels, labels_cam=None):
        assert len(labels) == len(self)

        # update self.img_items
        img_items = []
        if labels_cam is not None:
            for i in range(len(self)):
                img_items.append(
                    (self.img_items[i][0], labels[i], self.img_items[i][2], labels_cam[i]))
        else:
            for i in range(len(self)):
                img_items.append(
                    (self.img_items[i][0], labels[i], self.img_items[i][2]))
        self.img_items = img_items

        # update self.pids
        self.pids = list(set(labels))
        # update self.pid_dict
        self.pid_dict = {p: i for i, p in enumerate(self.pids)}
        self.pids_cam = labels_cam
        # self.pid_cam_dict = {p: i for i, p in enumerate(list(set(labels_cam)))}
