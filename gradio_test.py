from diffusers import (
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetSceneTextErasingPipeline,
    )
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os

import gradio as gr

os.environ["CUDA_VISIBLE_DEVICES"]="1"

pipe = StableDiffusionControlNetSceneTextErasingPipeline.from_pretrained('controlnet_scenetext_eraser/')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(torch.device('cuda:1'))

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda:1").manual_seed(1)


def inf(image, mask_image):
    
    image = Image.fromarray(image).resize((512, 512))
    mask_image = Image.fromarray(mask_image).resize((512, 512))

    image = pipe(
        image,
        mask_image,
        [mask_image],
        num_inference_steps=20,
        generator=generator,
        controlnet_conditioning_scale=1.0,
        guidance_scale=1.0
    ).images[0]

    return np.array(image)



if __name__ == "__main__":

    demo = gr.Interface(
    inf, 
    inputs=[gr.Image(), gr.Image()], 
    outputs="image",
    title="Scene Text Erasing, IIT-Jodhpur",
    )
    demo.launch(share=True)