# Authors: Hui Ren (rhfeiyang.github.io)
import torch
from PIL import Image
import argparse
import os, json, random

import matplotlib.pyplot as plt
import glob, re

from tqdm import tqdm
import numpy as np

import sys
import gc
from transformers import CLIPTextModel, CLIPTokenizer, BertModel, BertTokenizer

# import train_util

from utils.train_util import get_noisy_image, encode_prompts

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler

from typing import Any, Dict, List, Optional, Tuple, Union
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import argparse
# from diffusers.training_utils import EMAModel
import shutil
import yaml
from easydict import EasyDict
from utils.metrics import StyleContentMetric
from torchvision import transforms

from custom_datasets.coco import CustomCocoCaptions
from custom_datasets.imagepair import ImageSet
from custom_datasets import get_dataset
# from stable_diffusion.utils.modules import get_diffusion_modules
# from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor
import pickle
import time
def flush():
    torch.cuda.empty_cache()
    gc.collect()

def get_train_method(lora_weight):
    if lora_weight is None:
        return 'None'
    if 'full' in lora_weight:
        train_method = 'full'
    elif "down_1_up_2_attn" in lora_weight:
        train_method = 'up_2_attn'
        print(f"Using up_2_attn for {lora_weight}")
    elif "down_2_up_1_up_2_attn" in lora_weight:
        train_method = 'down_2_up_2_attn'
    elif "down_2_up_2_attn" in lora_weight:
        train_method = 'down_2_up_2_attn'
    elif "down_2_attn" in lora_weight:
        train_method = 'down_2_attn'
    elif 'noxattn' in lora_weight:
        train_method = 'noxattn'
    elif "xattn" in lora_weight:
        train_method = 'xattn'
    elif  "attn" in lora_weight:
        train_method = 'attn'
    elif "all_up" in lora_weight:
        train_method = 'all_up'
    else:
        train_method = 'None'
    return train_method

def get_validation_dataloader(infer_prompts:list[str]=None, infer_images :list[str]=None,resolution=512, batch_size=10, num_workers=4, val_set="laion_pop500"):
    data_transforms = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
    )
    def preprocess(example):
        ret={}
        ret["image"] = data_transforms(example["image"]) if "image" in example else None
        if "caption" in example:
            if isinstance(example["caption"][0], list):
                ret["caption"] = example["caption"][0][0]
            else:
                ret["caption"] = example["caption"][0]
        if "seed" in example:
            ret["seed"] = example["seed"]
        if "id" in example:
            ret["id"] = example["id"]
        if "path" in example:
            ret["path"] = example["path"]
        return ret

    def collate_fn(examples):
        out = {}
        if "image" in examples[0]:
            pixel_values = [example["image"] for example in examples]
            out["pixel_values"] = pixel_values
        # notice: only take the first prompt for each image
        if "caption" in examples[0]:
            prompts = [example["caption"] for example in examples]
            out["prompts"] = prompts
        if "seed" in examples[0]:
            seeds = [example["seed"] for example in examples]
            out["seed"] = seeds
        if "path" in examples[0]:
            paths = [example["path"] for example in examples]
            out["path"] = paths
        return out
    if infer_prompts is None:
        if val_set == "lhq500":
            dataset = get_dataset("lhq_sub500", get_val=False)["train"]
        elif val_set == "custom_coco100":
            dataset = get_dataset("custom_coco100", get_val=False)["train"]
        elif val_set == "custom_coco500":
            dataset = get_dataset("custom_coco500", get_val=False)["train"]

        elif os.path.isdir(val_set):
            image_folder = os.path.join(val_set, "paintings")
            caption_folder = os.path.join(val_set, "captions")
            dataset = ImageSet(folder=image_folder, caption=caption_folder, keep_in_mem=True)
        elif "custom_caption" in val_set:
            from custom_datasets.custom_caption import Caption_set
            name = val_set.replace("custom_caption_", "")
            dataset = Caption_set(set_name = name)
        elif val_set == "laion_pop500":
            dataset = get_dataset("laion_pop500", get_val=False)["train"]
        elif val_set == "laion_pop500_first_sentence":
            dataset = get_dataset("laion_pop500_first_sentence", get_val=False)["train"]
        else:
            raise ValueError("Unknown dataset")
        dataset.with_transform(preprocess)
    elif isinstance(infer_prompts, torch.utils.data.Dataset):
        dataset = infer_prompts
        try:
            dataset.with_transform(preprocess)
        except:
            pass

    else:
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, prompts, images=None):
                self.prompts = prompts
                self.images = images
                self.get_img = False
                if images is not None:
                    assert len(prompts) == len(images)
                    self.get_img = True
                    if isinstance(images[0], str):
                        self.images = [Image.open(image).convert("RGB") for image in images]
                else:
                    self.images = [None] * len(prompts)
            def __len__(self):
                return len(self.prompts)
            def __getitem__(self, idx):
                img = self.images[idx]
                if self.get_img and img is not None:
                    img = data_transforms(img)
                return {"caption": self.prompts[idx], "image":img}
        dataset = Dataset(infer_prompts, infer_images)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False,
                                             num_workers=num_workers, pin_memory=True)
    return dataloader

def get_lora_network(unet , lora_path, train_method="None", rank=1, alpha=1.0, device="cuda", weight_dtype=torch.float32):
    if train_method in [None, "None"]:
        train_method = get_train_method(lora_path)
        print(f"Train method: {train_method}")

    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'

    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    alpha = 1
    if "rank" in lora_path:
        rank = int(re.search(r'rank(\d+)', lora_path).group(1))
    if 'alpha1' in lora_path:
        alpha = 1.0
    print(f"Rank: {rank}, Alpha: {alpha}")

    network = LoRANetwork(
        unet,
        rank=rank,
        multiplier=1.0,
        alpha=alpha,
        train_method=train_method,
    ).to(device, dtype=weight_dtype)
    if lora_path not in [None, "None"]:
        lora_state_dict = torch.load(lora_path)
        miss = network.load_state_dict(lora_state_dict, strict=False)
        print(f"Missing: {miss}")
    ret = {"network": network, "train_method": train_method}
    return ret

def get_model(pretrained_ckpt_path, unet_ckpt=None,revision=None, variant=None, lora_path=None, weight_dtype=torch.float32,
              device="cuda"):
    modules = {}
    pipe = DiffusionPipeline.from_pretrained(pretrained_ckpt_path, revision=revision, variant=variant)
    if unet_ckpt is not None:
        pipe.unet.from_pretrained(unet_ckpt, subfolder="unet_ema", revision=revision, variant=variant)
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    modules["unet"] = unet
    modules["vae"] = vae
    modules["text_encoder"] = text_encoder
    modules["tokenizer"] = tokenizer
    # tokenizer = modules["tokenizer"]

    unet.enable_xformers_memory_efficient_attention()
    unet.to(device, dtype=weight_dtype)
    if weight_dtype != torch.bfloat16:
        vae.to(device, dtype=torch.float32)
    else:
        vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    if lora_path is not None:
        network = get_lora_network(unet, lora_path, device=device, weight_dtype=weight_dtype)
        modules["network"] = network
    return modules



@torch.no_grad()
def inference(network: LoRANetwork, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, vae: AutoencoderKL, unet: UNet2DConditionModel, noise_scheduler: LMSDiscreteScheduler,
              dataloader, height:int, width:int, scales:list = np.linspace(0,2,5),save_dir:str=None, seed:int = None,
              weight_dtype: torch.dtype = torch.float32, device: torch.device="cuda", batch_size:int=1, steps:int=50, guidance_scale:float=7.5, start_noise:int=800,
              uncond_prompt:str=None, uncond_embed=None, style_prompt = None, show:bool = False, no_load:bool=False, from_scratch=False):
    print(f"save dir: {save_dir}")
    if start_noise < 0:
        assert from_scratch
    network = network.eval()
    unet = unet.eval()
    vae = vae.eval()
    do_convert = not from_scratch

    if not do_convert:
        try:
            dataloader.dataset.get_img = False
        except:
            pass
        scales = list(scales)
    else:
        scales = ["Real Image"] + list(scales)

    if not no_load and os.path.exists(os.path.join(save_dir, "infer_imgs.pickle")):
        with open(os.path.join(save_dir, "infer_imgs.pickle"), 'rb') as f:
            pred_images = pickle.load(f)
        take=True
        for key in scales:
            if key not in pred_images:
                take=False
                break
        if take:
            print(f"Found existing inference results in {save_dir}", flush=True)
            return pred_images

    max_length = tokenizer.model_max_length

    pred_images = {scale :[] for scale in scales}
    all_seeds = {scale:[] for scale in scales}

    prompts = []
    ori_prompts = []
    if save_dir is not None:
        img_output_dir = os.path.join(save_dir, "outputs")
        os.makedirs(img_output_dir, exist_ok=True)

    if uncond_embed is None:
        if uncond_prompt is None:
            uncond_input_text = [""]
        else:
            uncond_input_text = [uncond_prompt]
        uncond_embed = encode_prompts(tokenizer = tokenizer, text_encoder = text_encoder, prompts = uncond_input_text)


    for batch in dataloader:
        ori_prompt = batch["prompts"]
        image = batch["pixel_values"] if do_convert else None
        if do_convert:
            pred_images["Real Image"] += image
        if isinstance(ori_prompt, list):
            if isinstance(text_encoder, CLIPTextModel):
                # trunc prompts for clip encoder
                ori_prompt = [p.split(".")[0]+"." for p in ori_prompt]
            prompt = [f"{p.strip()[::-1].replace('.', '',1)[::-1]} in the style of {style_prompt}" for p in ori_prompt] if style_prompt is not None else ori_prompt
        else:
            if isinstance(text_encoder, CLIPTextModel):
                ori_prompt = ori_prompt.split(".")[0]+"."
            prompt = f"{prompt.strip()[::-1].replace('.', '',1)[::-1]} in the style of {style_prompt}" if style_prompt is not None else ori_prompt

        bcz = len(prompt)
        single_seed = seed
        if dataloader.batch_size == 1 and seed is None:
            if "seed" in batch:
                single_seed = batch["seed"][0]

        print(f"{prompt}, seed={single_seed}")

        # text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        # original_embeddings = text_encoder(**text_input)[0]

        prompts += prompt
        ori_prompts += ori_prompt
        # style_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        # # style_embeddings = text_encoder(**style_input)[0]
        # style_embeddings = text_encoder(style_input.input_ids, return_dict=False)[0]

        style_embeddings = encode_prompts(tokenizer = tokenizer, text_encoder = text_encoder, prompts = prompt)
        original_embeddings = encode_prompts(tokenizer = tokenizer, text_encoder = text_encoder, prompts = ori_prompt)
        if uncond_embed.shape[0] == 1 and bcz > 1:
            uncond_embeddings = uncond_embed.repeat(bcz, 1, 1)
        else:
            uncond_embeddings = uncond_embed
        style_text_embeddings = torch.cat([uncond_embeddings, style_embeddings])
        original_embeddings = torch.cat([uncond_embeddings, original_embeddings])

        generator = torch.manual_seed(single_seed) if single_seed is not None else None
        noise_scheduler.set_timesteps(steps)
        if do_convert:
            noised_latent, _, _ = get_noisy_image(image, vae, generator, unet, noise_scheduler, total_timesteps=int((1000-start_noise)/1000 *steps))
        else:
            latent_shape =  (bcz, 4, height//8, width//8)
            noised_latent = randn_tensor(latent_shape, generator=generator, device=vae.device)
        noised_latent = noised_latent.to(unet.dtype)
        noised_latent = noised_latent * noise_scheduler.init_noise_sigma
        for scale in scales:
            start_time = time.time()
            if not isinstance(scale, float) and not isinstance(scale, int):
                continue

            latents = noised_latent.clone().to(weight_dtype).to(device)
            noise_scheduler.set_timesteps(steps)
            for t in tqdm(noise_scheduler.timesteps):
                if do_convert and t>start_noise:
                    continue
                else:
                    if t > start_noise and start_noise >= 0:
                        current_scale = 0
                    else:
                        current_scale = scale
                network.set_lora_slider(scale=current_scale)
                text_embedding = style_text_embeddings
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
                # predict the noise residual
                with network:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embedding).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if isinstance(noise_scheduler, DDPMScheduler):
                    latents = noise_scheduler.step(noise_pred, t, latents, generator=torch.manual_seed(single_seed+t) if single_seed is not None else None).prev_sample
                else:
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents.to(vae.dtype)


            with torch.no_grad():
                image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")


            pil_images = [Image.fromarray(image) for image in images]
            pred_images[scale]+=pil_images
            all_seeds[scale] += [single_seed] * bcz

            end_time = time.time()
            print(f"Time taken for one batch, Art Adapter scale={scale}: {end_time-start_time}", flush=True)

        if save_dir is not None or show:
            end_idx = len(list(pred_images.values())[0])
            for i in range(end_idx-bcz, end_idx):
                keys = list(pred_images.keys())
                images_list = [pred_images[key][i] for key in keys]
                prompt = prompts[i]
                if len(scales)==1:
                    plt.imshow(images_list[0])
                    plt.axis('off')
                    plt.title(f"{prompt}_{single_seed}_start{start_noise}", fontsize=20)
                else:
                    fig, ax = plt.subplots(1, len(images_list), figsize=(len(scales)*5,6), layout="constrained")
                    for id, a in enumerate(ax):
                        a.imshow(images_list[id])
                        if isinstance(scales[id], float) or isinstance(scales[id], int):
                            a.set_title(f"Art Adapter scale={scales[id]}", fontsize=20)
                        else:
                            a.set_title(f"{keys[id]}", fontsize=20)
                        a.axis('off')

                    # plt.suptitle(f"{os.path.basename(lora_weight).replace('.pt','')}", fontsize=20)

                    # plt.tight_layout()
                    # if do_convert:
                    #     plt.suptitle(f"{prompt}\nseed{single_seed}_start{start_noise}_guidance{guidance_scale}", fontsize=20)
                    # else:
                    #     plt.suptitle(f"{prompt}\nseed{single_seed}_from_scratch_guidance{guidance_scale}", fontsize=20)

                if save_dir is not None:
                    plt.savefig(f"{img_output_dir}/{prompt.replace(' ', '_')[:100]}_seed{single_seed}_start{start_noise}.png")
                if show:
                    plt.show()
                plt.close()

        flush()

    if save_dir is not None:
        with open(os.path.join(save_dir, "infer_imgs.pickle" ), 'wb') as f:
            pickle.dump(pred_images, f)
        with open(os.path.join(save_dir, "all_seeds.pickle"), 'wb') as f:
            to_save={"all_seeds":all_seeds, "batch_size":batch_size}
            pickle.dump(to_save, f)
        for scale, images in pred_images.items():
            subfolder = os.path.join(save_dir,"images", f"{scale}")
            os.makedirs(subfolder, exist_ok=True)

            used_prompt = ori_prompts
            if (isinstance(scale, float) or isinstance(scale, int)): #and scale != 0:
                used_prompt = prompts
            for i, image in enumerate(images):
                if scale == "Real Image":
                    suffix = ""
                else:
                    suffix = f"_seed{all_seeds[scale][i]}"
                image.save(os.path.join(subfolder, f"{used_prompt[i].replace(' ', '_')[:100]}{suffix}.jpg"))
        with open(os.path.join(save_dir, "infer_prompts.txt"), 'w') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        with open(os.path.join(save_dir, "ori_prompts.txt"), 'w') as f:
            for prompt in ori_prompts:
                f.write(f"{prompt}\n")
        print(f"Saved inference results to {save_dir}", flush=True)
    return pred_images, prompts

@torch.no_grad()
def infer_metric(ref_image_folder,pred_images, prompts, save_dir, start_noise=""):
    prompts = [prompt.split(" in the style of ")[0] for prompt in prompts]
    scores = {}
    original_images = pred_images["Real Image"] if "Real Image" in pred_images else None
    metric = StyleContentMetric(ref_image_folder)
    for scale, images in pred_images.items():
        score = metric(images, original_images, prompts)

        scores[scale] = score
        print(f"Style transfer score at scale {scale}: {score}")
    scores["ref_path"] = ref_image_folder
    save_name = f"scores_start{start_noise}.json"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, save_name), 'w') as f:
        json.dump(scores, f, indent=2)
    return scores

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with LoRA')
    parser.add_argument('--lora_weights', type=str, default=["None"],
                        nargs='+', help='path to your model file')
    parser.add_argument('--prompts', type=str, default=[],
                        nargs='+', help='prompts to try')
    parser.add_argument("--prompt_file", type=str, default=None, help="path to the prompt file")
    parser.add_argument("--prompt_file_key", type=str, default="prompts", help="key to the prompt file")
    parser.add_argument('--resolution', type=int, default=512, help='resolution of the image')
    parser.add_argument('--seed', type=int, default=None, help='seed for the random number generator')
    parser.add_argument("--start_noise", type=int, default=800, help="start noise")
    parser.add_argument("--from_scratch", default=False, action="store_true", help="from scratch")
    parser.add_argument("--ref_image_folder", type=str, default=None, help="folder containing reference images")
    parser.add_argument("--show", action="store_true", help="show the image")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--scales", type=float, default=[0.,1.], nargs='+', help="scales to test")
    parser.add_argument("--train_method", type=str, default=None, help="train method")

    # parser.add_argument("--vae_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Path to the VAE model.")
    # parser.add_argument("--text_encoder_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Path to the text encoder model.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="rhfeiyang/art-free-diffusion-v1", help="Path to the pretrained model.")
    parser.add_argument("--unet_ckpt", default=None, type=str, help="Path to the unet checkpoint")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="guidance scale")
    parser.add_argument("--infer_mode", default="sks_art",  help="inference mode") #, choices=["style", "ori", "artist", "sks_art","Peter"]
    parser.add_argument("--save_dir", type=str, default="inference_output", help="save directory")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--no_load", action="store_true", help="no load the pre-inferred results")
    parser.add_argument("--infer_prompts", type=str, default=None, nargs="+", help="prompts to infer")
    parser.add_argument("--infer_images", type=str, default=None, nargs="+", help="images to infer")
    parser.add_argument("--rank", type=int, default=1, help="rank of the lora")
    parser.add_argument("--val_set", type=str, default="laion_pop500",  help="validation set")
    parser.add_argument("--folder_name", type=str, default=None, help="folder name")
    parser.add_argument("--scheduler_type",type=str, choices=["ddpm", "ddim", "pndm","lms"], default="ddpm", help="scheduler type")
    parser.add_argument("--infer_steps", type=int, default=50, help="inference steps")
    parser.add_argument("--weight_dtype", type=str, default="fp32", help="weight dtype")
    parser.add_argument("--custom_coco_cap", action="store_true", help="use custom coco caption")
    args = parser.parse_args()
    if args.infer_prompts is not None and len(args.infer_prompts) == 1 and os.path.isfile(args.infer_prompts[0]):
        if args.infer_prompts[0].endswith(".txt") and args.custom_coco_cap:
            args.infer_prompts = CustomCocoCaptions(custom_file=args.infer_prompts[0])
        elif args.infer_prompts[0].endswith(".txt"):
            with open(args.infer_prompts[0], 'r') as f:
                args.infer_prompts = f.readlines()
                args.infer_prompts = [prompt.strip() for prompt in args.infer_prompts]
        elif args.infer_prompts[0].endswith(".csv"):
            from custom_datasets.custom_caption import Caption_set
            caption_set = Caption_set(args.infer_prompts[0])
            args.infer_prompts = caption_set


    if args.infer_mode == "style":
        with open(os.path.join(args.ref_image_folder, "style_label.txt"), 'r') as f:
            args.style_label = f.readlines()[0].strip()
    elif args.infer_mode == "artist":
        with open(os.path.join(args.ref_image_folder, "style_label.txt"), 'r') as f:
            args.style_label = f.readlines()[0].strip()
            args.style_label = args.style_label.split(",")[0].strip()
    elif args.infer_mode == "ori":
        args.style_label = None
    else:
        args.style_label = args.infer_mode.replace("_", " ")
    if args.ref_image_folder is not None:
        args.ref_image_folder = os.path.join(args.ref_image_folder, "paintings")

    if args.start_noise < 0:
        args.from_scratch = True


    print(args.__dict__)
    return args


def main(args):
    lora_weights = args.lora_weights

    if len(lora_weights) == 1 and isinstance(lora_weights[0], str) and os.path.isdir(lora_weights[0]):
        lora_weights = glob.glob(os.path.join(lora_weights[0], "*.pt"))
        lora_weights=sorted(lora_weights, reverse=True)

    width = args.resolution
    height = args.resolution
    steps = args.infer_steps

    revision = None
    device = 'cuda'
    rank = args.rank
    if args.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif args.weight_dtype=="fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype=="bf16":
        weight_dtype = torch.bfloat16

    modules = get_model(args.pretrained_model_name_or_path, unet_ckpt=args.unet_ckpt, revision=revision, variant=None, lora_path=None, weight_dtype=weight_dtype, device=device, )
    if args.scheduler_type == "pndm":
        noise_scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    elif args.scheduler_type == "ddpm":
        noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    elif args.scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type="epsilon",
        )
    elif args.scheduler_type == "lms":
        noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                             beta_end=0.012,
                             beta_schedule="scaled_linear",
                             num_train_timesteps=1000)
    else:
        raise ValueError("Unknown scheduler type")
    cache=EasyDict()
    cache.modules = modules

    unet = modules["unet"]
    vae = modules["vae"]
    text_encoder = modules["text_encoder"]
    tokenizer = modules["tokenizer"]

    unet.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    ## dataloader
    dataloader = get_validation_dataloader(infer_prompts=args.infer_prompts, infer_images=args.infer_images,
                                           resolution=args.resolution,
                                           batch_size=args.batch_size, num_workers=args.num_workers,
                                           val_set=args.val_set)


    for lora_weight in lora_weights:
        print(f"Testing {lora_weight}")
        # for different seeds on same prompt
        seed = args.seed

        network_ret = get_lora_network(unet, lora_weight, train_method=args.train_method, rank=rank, alpha=1.0, device=device, weight_dtype=weight_dtype)
        network = network_ret["network"]
        train_method = network_ret["train_method"]
        if args.save_dir is not None:
            save_dir = args.save_dir
            if args.style_label is not None:
                save_dir = os.path.join(save_dir, f"{args.style_label.replace(' ', '_')}")
            else:
                save_dir = os.path.join(save_dir, f"ori/{args.start_noise}")
        else:
            if args.folder_name is not None:
                folder_name = args.folder_name
            else:
                folder_name = "validation" if args.infer_prompts is None else "validation_prompts"
            save_dir = os.path.join(os.path.dirname(lora_weight), f"{folder_name}/{train_method}", os.path.basename(lora_weight).replace('.pt','').split('_')[-1])
        if args.infer_prompts is None:
            save_dir = os.path.join(save_dir, f"{args.val_set}")

        infer_config = f"{args.scheduler_type}{args.infer_steps}_{args.weight_dtype}_guidance{args.guidance_scale}"
        save_dir = os.path.join(save_dir, infer_config)
        os.makedirs(save_dir, exist_ok=True)
        if args.from_scratch:
            save_dir = os.path.join(save_dir, "from_scratch")
        else:
            save_dir = os.path.join(save_dir, "transfer")
        save_dir = os.path.join(save_dir, f"start{args.start_noise}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "infer_args.yaml"), 'w') as f:
            yaml.dump(vars(args), f)
        # save code
        code_dir = os.path.join(save_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        current_file = os.path.basename(__file__)
        shutil.copy(__file__, os.path.join(code_dir, current_file))
        with torch.no_grad():
            pred_images, prompts = inference(network, tokenizer, text_encoder, vae, unet, noise_scheduler, dataloader, height, width,
                                    args.scales, save_dir, seed, weight_dtype, device, args.batch_size, steps, guidance_scale=args.guidance_scale,
                                    start_noise=args.start_noise, show=args.show, style_prompt=args.style_label, no_load=args.no_load,
                                    from_scratch=args.from_scratch)

            if args.ref_image_folder is not None:
                flush()
                print("Calculating metrics")
                infer_metric(args.ref_image_folder, pred_images, save_dir, args.start_noise)

if __name__ == "__main__":
    args = parse_args()
    main(args)