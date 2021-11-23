import torch
import torch.nn.functional as F
from ..builder import REIDS, build_backbone, build_head, build_neck
from .base import BaseModel
from .baseline import Baseline
from ..utils import GatherLayer


@REIDS.register_module()
class Hybrid(Baseline):

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'
        batch_size = img.shape[0]
        img = torch.cat(torch.unbind(img, dim=1), dim=0)
        z = self.neck(self.backbone(img))[0]
        z_ori, z_aug = torch.split(z, [batch_size, 2 * batch_size], dim=0)
        z_aug = F.normalize(z_aug, dim=1)
        z1, z2 = torch.split(z_aug, [batch_size, batch_size], dim=0)
        z_aug = torch.cat((z1.unsqueeze(1), z2.unsqueeze(1)), dim=1)
        z_aug = torch.cat(GatherLayer.apply(z_aug), dim=0)

        losses = self.head(z_ori, z_aug, **kwargs)

        return losses
