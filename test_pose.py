from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from PIL import Image

controlnet = [
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16),
    # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),r
]

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
        feature_extractor=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

prompt = "man holding sign that says 'नमस्ते',  High quality "
# negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.Generator(device="cpu").manual_seed(1)

images = Image.open("/DATA/ocr_team_2/onkar2/diffusers/download_pose.png")

image = pipe(
    prompt,
    [images],
    num_inference_steps=30,
    generator=generator,
    # negative_prompt=negative_prompt,
    controlnet_conditioning_scale=[1.0],
).images[0]

image.save('test.png')


