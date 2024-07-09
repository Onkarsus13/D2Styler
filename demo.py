import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from power_paint_tokenizer import PowerPaintTokenizer
from diffusers.utils import load_image


def add_task_to_prompt(prompt, negative_prompt, task):
    if task == "object-removal":
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    elif task == "shape-guided":
        promptA = prompt + " P_shape"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif task == "image-outpainting":
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    else:
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt

    return promptA, promptB, negative_promptA, negative_promptB


@torch.inference_mode()
def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    negative_prompt,
    task,
):
    width, height = input_image["image"].convert("RGB").size

    if width < height:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((640, int(height / width * 640)))
        )
    else:
        input_image["image"] = (
            input_image["image"].convert("RGB").resize((int(width / height * 640), 640))
        )

    promptA, promptB, negative_promptA, negative_promptB = add_task_to_prompt(
        prompt, negative_prompt, task
    )
    print(promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))
    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image["image"].convert("RGB"),
        mask_image=input_image["mask"].convert("RGB"),
        width=H,
        height=W,
        guidance_scale=scale,
        num_inference_steps=ddim_steps,
    ).images[0]
    mask_np = np.array(input_image["mask"].convert("RGB"))
    red = np.array(result).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
            + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )
    m_img = (
        input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    )
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image["mask"].convert("RGB"), result_m]

    dict_out = [input_image["image"].convert("RGB"), result_paste]

    return dict_out, dict_res


pipe = Pipeline.from_pretrained(
    "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
    variant="fp16",
)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
pipe.tokenizer = PowerPaintTokenizer(pipe.tokenizer)
pipe = pipe.to(device)


fold = "dogs-jump"
img = "00030.jpg"
image = Image.open(f"/data2/onkar/sktech_diff_data/First_Exp_Codes/train/all_images/{fold}/{img}").resize((512, 512))
mask = Image.open("./mask1.png").resize((512, 512))

input_image = {"image": image, "mask": mask}

prompt = "polar bear seating on grass"
negative_prompt = None#"out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
fitting_degree = 1
ddim_steps = 30





dict_out, dict_res = predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    16.5,
    negative_prompt,
    "inpaint",
)

result_image = dict_out[1]
result_image.save("test_1.png")