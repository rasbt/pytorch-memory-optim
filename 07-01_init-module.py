import gc
import psutil

import torch
from lightning.fabric import Fabric

from torchvision.models import vit_l_16
from torchvision.models import ViT_L_16_Weights
from watermark import watermark


if __name__ == "__main__":
    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    

    print("Without Fabric")

    cpu_memory_before = psutil.Process().memory_info().rss / (1024**3)

    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    model.to(torch.device("cuda")).to(torch.float16)

    cpu_ram_after = psutil.Process().memory_info().rss / (1024**3)

    print(f"CPU Memory used: {cpu_ram_after - cpu_memory_before / 1e9:.02f} GB")
    print(f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")