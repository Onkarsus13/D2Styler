from omegaconf import OmegaConf
from unet import UNet3DConditionModel

config = OmegaConf.load("/home/awd8324/onkar/AnimateDiff/configs/training/v1/training.yaml")

print(config)
unet = UNet3DConditionModel.from_pretrained_2d(
            "runwayml/stable-diffusion-v1-5", subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs)
        )

