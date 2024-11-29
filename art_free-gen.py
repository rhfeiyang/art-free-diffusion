from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
import torch

pipe = DiffusionPipeline.from_pretrained("rhfeiyang/art-free-diffusion-v1",).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
prompt = "The image depicts a picturesque small town by a river, featuring several docked boats. Surrounded by trees, the town is near a large body of water, highlighting its popularity for boating and water activities. The serene composition, with trees and boats, underscores the town's natural beauty and tranquil charm."
images = pipe(prompt,
              num_inference_steps=50, guidance_scale=7.5, generator = torch.Generator().manual_seed(0)
              ).images
plt.imshow(images[0])
plt.title(prompt)
# plt.axis('off')
plt.show()
