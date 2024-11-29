from typing import Optional, Union

import torch

from transformers import CLIPTextModel, CLIPTokenizer, BertModel, BertTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.image_processor import VaeImageProcessor
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# from imagesliders.model_util import SDXL_TEXT_ENCODER_TYPE
from diffusers.utils.torch_utils import randn_tensor

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

SDXL_TEXT_ENCODER_TYPE = Union[CLIPTextModel, CLIPTextModelWithProjection]

from tqdm import tqdm

UNET_IN_CHANNELS = 4  # Stable Diffusion  in_channels
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8

UNET_ATTENTION_TIME_EMBED_DIM = 256  # XL
TEXT_ENCODER_2_PROJECTION_DIM = 1280
UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM = 2816


def get_random_noise(
    batch_size: int, height: int, width: int, generator: torch.Generator = None
) -> torch.Tensor:
    return torch.randn(
        (
            batch_size,
            UNET_IN_CHANNELS,
            height // VAE_SCALE_FACTOR,
            width // VAE_SCALE_FACTOR,
        ),
        generator=generator,
        device="cpu",
    )



def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn(
        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    )
    return latents


def get_initial_latents(
    scheduler: SchedulerMixin,
    n_imgs: int,
    height: int,
    width: int,
    n_prompts: int,
    generator=None,
) -> torch.Tensor:
    noise = get_random_noise(n_imgs, height, width, generator=generator).repeat(
        n_prompts, 1, 1, 1
    )

    latents = noise * scheduler.init_noise_sigma

    return latents


def text_tokenize(
    tokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts,
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


def text_encode(text_encoder , tokens):
    tokens = tokens.to(text_encoder.device)
    if isinstance(text_encoder, BertModel):
        embed = text_encoder(**tokens, return_dict=False)[0]
    elif isinstance(text_encoder, CLIPTextModel):
        # embed = text_encoder(**tokens, return_dict=False)[0]
        embed = text_encoder(tokens.input_ids, return_dict=False)[0]
    else:
        raise ValueError("text_encoder must be BertModel or CLIPTextModel")
    return embed

def encode_prompts(
    tokenizer,
    text_encoder,
    prompts: list[str],
):
    # print(f"prompts: {prompts}")
    text_tokens = text_tokenize(tokenizer, prompts)
    # print(f"text_tokens: {text_tokens}")
    text_embeddings = text_encode(text_encoder, text_tokens)
    # print(f"text_embeddings: {text_embeddings}")
    

    return text_embeddings

def prompt_replace(original, key="{prompt}", prompt=""):
    if key not in original:
        return original

    if isinstance(prompt, list):
        ret =[]
        for p in prompt:
            p = p.replace(".", "")
            r = original.replace(key, p)
            r = r.capitalize()
            ret.append(r)
    else:
        prompt = prompt.replace(".", "")
        ret = original.replace(key, prompt)
        ret = ret.capitalize()
    return ret



def text_encode_xl(
    text_encoder: SDXL_TEXT_ENCODER_TYPE,
    tokens: torch.FloatTensor,
    num_images_per_prompt: int = 1,
):
    prompt_embeds = text_encoder(
        tokens.to(text_encoder.device), output_hidden_states=True
    )
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer

    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompts_xl(
    tokenizers: list[CLIPTokenizer],
    text_encoders: list[SDXL_TEXT_ENCODER_TYPE],
    prompts: list[str],
    num_images_per_prompt: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # text_encoder and text_encoder_2's penuultimate layer's output
    text_embeds_list = []
    pooled_text_embeds = None  # always text_encoder_2's pool

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_tokens_input_ids = text_tokenize(tokenizer, prompts)
        text_embeds, pooled_text_embeds = text_encode_xl(
            text_encoder, text_tokens_input_ids, num_images_per_prompt
        )

        text_embeds_list.append(text_embeds)

    bs_embed = pooled_text_embeds.shape[0]
    pooled_text_embeds = pooled_text_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )

    return torch.concat(text_embeds_list, dim=-1), pooled_text_embeds


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    if conditional.shape[0] == n_imgs and unconditional.shape[0] == 1:
        return torch.cat([unconditional.repeat(n_imgs, 1, 1), conditional], dim=0)
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)


def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
    # batch_size = latents.shape[0]
    # text_embeddings = text_embeddings.repeat_interleave(batch_size, dim=0)
    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target



@torch.no_grad()
def diffusion(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,
    total_timesteps: int = 1000,
    start_timesteps=0,
    **kwargs,
):
    # latents_steps = []

    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        noise_pred = predict_noise(
            unet, scheduler, timestep, latents, text_embeddings, **kwargs
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents

@torch.no_grad()
def get_noisy_image(
    img,
    vae,
    generator,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    total_timesteps: int = 1000,
    start_timesteps=0,
    
    **kwargs,
):
    # latents_steps = []
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    image = img
    # im_orig = image
    device = vae.device
    image = image_processor.preprocess(image).to(device)

    init_latents = vae.encode(image).latent_dist.sample(None)
    init_latents = vae.config.scaling_factor * init_latents

    init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape

    noise = randn_tensor(shape, generator=generator, device=device)

    time_ = total_timesteps
    timestep = scheduler.timesteps[time_:time_+1]
    # get latents
    noised_latents = scheduler.add_noise(init_latents, noise, timestep)
    
    return noised_latents, noise, init_latents

def subtract_noise(
        latent: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
        scheduler: SchedulerMixin,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
    # for the subsequent add_noise calls
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=latent.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=latent.dtype)
    timesteps = timesteps.to(latent.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(latent.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(latent.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    denoised_latent =  (latent - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    return denoised_latent
def get_denoised_image(
        latents: torch.FloatTensor,
        noise_pred: torch.FloatTensor,
        timestep: int,
        # total_timesteps: int,
        scheduler: SchedulerMixin,
        vae: VaeImageProcessor,
):
    denoised_latents = subtract_noise(latents, noise_pred, timestep, scheduler)
    denoised_latents = denoised_latents / vae.config.scaling_factor # 0.18215
    denoised_img = vae.decode(denoised_latents).sample
    # denoised_img = denoised_img.clamp(-1,1)
    return denoised_img


def rescale_noise_cfg(
    noise_cfg: torch.FloatTensor, noise_pred_text, guidance_rescale=0.0
):

    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )

    return noise_cfg


def predict_noise_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    add_text_embeddings: torch.FloatTensor,  # pooled なやつ
    add_time_ids: torch.FloatTensor,
    guidance_scale=7.5,
    guidance_rescale=0.7,
) -> torch.FloatTensor:
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    added_cond_kwargs = {
        "text_embeds": add_text_embeddings,
        "time_ids": add_time_ids,
    }

    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    noise_pred = rescale_noise_cfg(
        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
    )

    return guided_target


@torch.no_grad()
def diffusion_xl(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    latents: torch.FloatTensor,
    text_embeddings: tuple[torch.FloatTensor, torch.FloatTensor],
    add_text_embeddings: torch.FloatTensor,
    add_time_ids: torch.FloatTensor,
    guidance_scale: float = 1.0,
    total_timesteps: int = 1000,
    start_timesteps=0,
):
    # latents_steps = []

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps]):
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            text_embeddings,
            add_text_embeddings,
            add_time_ids,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # return latents_steps
    return latents


# for XL
def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
):
    if dynamic_crops:
        # random float scale between 1 and 3
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        # random position
        crops_coords_top_left = (
            torch.randint(0, original_size[0] - height, (1,)).item(),
            torch.randint(0, original_size[1] - width, (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    # this is expected as 6
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    # this is expected as 2816
    passed_add_embed_dim = (
        UNET_ATTENTION_TIME_EMBED_DIM * len(add_time_ids)  # 256 * 6
        + TEXT_ENCODER_2_PROJECTION_DIM  # + 1280
    )
    if passed_add_embed_dim != UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM:
        raise ValueError(
            f"Model expects an added time embedding vector of length {UNET_PROJECTION_CLASS_EMBEDDING_INPUT_DIM}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt
            
            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width
