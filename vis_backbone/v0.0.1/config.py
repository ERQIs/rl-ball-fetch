from dataclasses import dataclass


@dataclass
class StageAConfig:
    image_size: int = 64
    feature_size: int = 8
    input_channels: int = 1
    channels: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    lambda_rec: float = 1.0
    lambda_trans: float = 1.0
    lambda_wd: float = 1.0
    lambda_nb: float = 0.0
    foreground_weight: float = 6.0
    foreground_dark_threshold: float = 0.55
    foreground_dilate_kernel: int = 5
