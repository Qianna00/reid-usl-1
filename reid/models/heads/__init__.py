from .hybrid_memory_head import HybridMemoryHead
from .latent_pred_head import LatentPredictHead
from .mmcl_head import MMCLHead
from .sup_contrast_head import SupContrastHead
from .scl_head import AnotherSCLHead
from .cam_aware_scl_head import CamAwareSCLHead, AnotherCamAwareSCLHead, AnotherNewCamAwareSCLHead
from .spcl_tmp_hybrid_head import HybridHead

__all__ = [
    'HybridMemoryHead', 'LatentPredictHead', 'MMCLHead', 'SupContrastHead',
    'AnotherSCLHead', 'CamAwareSCLHead', 'AnotherCamAwareSCLHead',
    'AnotherNewCamAwareSCLHead', 'HybridHead'
]
