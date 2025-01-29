from .erm import ERMTrainer
# from .lff import LfFTrainer
# from .sd import SDTrainer
# from .eiil import EIILTrainer
# from .jtt import JTTTrainer
# from .debian import DebiANTrainer
# from .wtm_aug import WatermarkAugTrainer
# from .bg_aug import BackgroundAugTrainer
# from .txt_aug import TextureAugTrainer
# from .lle import LLETrainer
# from .mixup import MixupTrainer
# from .augmix import AugMixTrainer
# from .cutmix import CutMixTrainer
# from .cutout import CutoutTrainer
from .sebra import OursTrainer

method_to_trainer = {
    "erm": ERMTrainer,
    "ours": OursTrainer
}
