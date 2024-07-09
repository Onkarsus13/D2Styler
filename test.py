import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, Optional
import PIL
import requests
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, PNDMScheduler, ControlNetModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline


if __name__ == "__main__":

    accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision='fp16',
            log_with='tensorboard',
            # logging_dir='logs',
        )


    pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2-inpainting'
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision='fp16',
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder='tokenizer'
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision="fp16",
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision='fp16',
    )

    controlnet = ControlNetModel.from_unet(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=0.0001,
    )


    noise_scheduler = PNDMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    epochs=100
    num_step_loader = 800
    lr_scheduler = get_scheduler(
            'cosine',
            optimizer=optimizer,
            num_warmup_steps=300,
            num_training_steps = epochs*num_step_loader
        )

    train_dataloader = None

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    max_train_step = 10000
    epochs = 50


    for epoch in range(epochs):
        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                _,_,h,w = batch["pixel_values"].shape()
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                mask_latent = vae.encode(batch['image_mask'].to(weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
                mask_latent = latent * 0.18215

                mask = batch['mask']
                mask = torch.nn.functional.interpolate(
                    mask, size=(h // vae_scale_factor, w // vae_scale_factor)
                )


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noisy_latents = torch.cat([noisy_latents, mask, mask_latent], axis=1)
                

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )
                
                model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            print(logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)

        p = StableDiffusionControlNetInpaintPipeline(
                vae= vae,
                text_encoder= text_encoder,
                tokenizer= tokenizer,
                unet= unet,
                scheduler=noise_scheduler,
                controlnet = controlnet,
                safety_checker=None,
                feature_extractor=None
        )
        p.save_pretrained('custom_SD_CN_Inpaint/')

    accelerator.end_training()

# def download_image(url):
#     response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert("RGB")

# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

# init_image = download_image(img_url).resize((512, 256))
# mask_image = download_image(mask_url).resize((512, 512))

# p = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "custom_SD_CN_Inpaint/",
#     revision="fp16",
#     torch_dtype=torch.float16,
# )
# p.to("cuda")

# prompt = "lion roaring on the park bench"
# image = p(prompt=prompt, image=init_image, mask_image=mask_image,  control_image=init_image, guidance_scale = 8.0, num_inference_steps = 60).images[0]
# image.save('gone.jpg')
