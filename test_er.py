from diffusers import (
    UniPCMultistepScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetSceneTextErasingPipeline,
    StableDiffusionInpaintPipeline,
    AutoPipelineForInpainting
    )
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import ceil
import glob

# Given JSON data
data_polygon={
    "boxes": [
        {
            "type": "polygon",
            "label": "scene-text",
            "x": "276.3569",
            "y": "180.4924",
            "width": "477.0107",
            "height": "288.7850",
            "points": [
                [
                    401.05263157894734,
                    54.73684210526315
                ],
                [
                    287.6089638157895,
                    73.51562499999997
                ],
                [
                    246.86266447368422,
                    101.55633223684207
                ],
                [
                    179.44284539473682,
                    124.18379934210526
                ],
                [
                    37.85156250000006,
                    175.546875
                ],
                [
                    42.87828947368422,
                    255.0657894736842
                ],
                [
                    127.77754934210532,
                    324.8848684210526
                ],
                [
                    268.1167763157895,
                    320.1480263157895
                ],
                [
                    331.4391447368421,
                    317.8865131578947
                ],
                [
                    430.540707236842,
                    278.7129934210526
                ],
                [
                    459.00082236842104,
                    209.04399671052633
                ],
                [
                    495.7524671052632,
                    154.01932565789474
                ],
                [
                    514.8622532894736,
                    63.20723684210529
                ],
                [
                    479.17968749999994,
                    36.09991776315792
                ],
                [
                    428.4703947368421,
                    39.90748355263159
                ]
            ],
            "color": "#C7FC00"
        }
    ],
    "height": 360,
    "key": "00020.jpg",
    "width": 640
}
# Create an empty image with the same size as the original image
mask_polygon = np.zeros((data_polygon["height"], data_polygon["width"]), dtype=np.uint8)

# Process each polygon to create a binary mask
for box in data_polygon["boxes"]:
    if box["type"] == "polygon":
        # Convert points to a format suitable for cv2.fillPoly
        pts = np.array(box["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask_polygon, [pts], color=255)  # Fill the polygon area with white


plt.imsave("./mask1.png", mask_polygon, cmap='gray')


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model_path = "Sanster/anything-4.0-inpainting"

pipe = AutoPipelineForInpainting.from_pretrained(model_path, 
                                                 torch_dtype=torch.float16, variant="fp16"
                                                 ).to(device)
pipe.to(device)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
generator = torch.Generator(device).manual_seed(12345612)

# fold = "dogs-jump"
# img = "00018.jpg"
# image = Image.open(f"/data2/onkar/sktech_diff_data/First_Exp_Codes/train/all_images/{fold}/{img}")

image = Image.open("/home/awd8324/onkar/lava_utils/results/00005.jpg")
# mask_image = Image.open(f"/data2/onkar/sktech_diff_data/First_Exp_Codes/train/all_masks/{fold}/{img.split('.')[0]+'.png' }")
mask_image = Image.open("mask1.png")


prompt = "cute butterfly, disney, magical, high quality"
image = pipe(prompt=prompt, image=image, mask_image=mask_image, num_inference_steps=30, guidance_scale=12.5, generator=generator, height=480, width=840).images[0]
image.save("./test_4.png")

# files = glob.glob(f"/data2/onkar/sktech_diff_data/First_Exp_Codes/train/all_images/{fold}/*.jpg")

# for f in files:
#     f_name = f.split('/')[-1]

#     image = Image.open(f)
#     image = pipe(prompt=prompt, image=image, mask_image=mask_image, num_inference_steps=30, guidance_scale=8.5, generator=generator).images[0]

#     image.save(f"results/{f_name}")

