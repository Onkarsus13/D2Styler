from diffusers import VQModel, Transformer2DModel, VQDiffusionScheduler
import torch
from sheduling_VQ import DiffusionTransformer
m = "microsoft/vq-diffusion-ithq"



vae = VQModel.from_pretrained(m, subfolder="vqvae")
trans = Transformer2DModel.from_pretrained(m, subfolder="transformer")
scheduler = VQDiffusionScheduler.from_pretrained(m, subfolder="scheduler")


x = torch.randn((2, 3, 256, 256))

en = vae.encode(x).latents
q = vae.quantize(en)[2][-1].view(2, -1)
# print(q.shape)
# print(q.min(), q.max())

timestep = torch.tensor([3, 4]).to(torch.int64).view(-1, 1)
print(timestep.shape)

te = trans(
    q,
    encoder_hidden_states=torch.randn(2, 77, 512),
    timestep=timestep,
    ).sample

opt = torch.optim.Adam(trans.parameters(), lr=0.0001)
trans.train()

diff = DiffusionTransformer(transformer=trans)
for _ in range(20):


    trans.zero_grad()
    xx = diff._train_loss(q, torch.randn(2, 49, 512))
    xx[-1].mean().backward()
    opt.step()
    print(xx[-1].mean().item())






