# Authors: Hui Ren (rhfeiyang.github.io)
# ref: https://github.com/rohitgandikota/sliders/blob/main/trainscripts/imagesliders/train_lora-scale.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm
import os, glob
import sys
sys.path.insert(0, "utils")

from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from diffusers import DDIMScheduler, DiffusionPipeline
import utils.train_util as train_util
import utils.model_util as model_util
import utils.debug_util as debug_util
# import prompt_util
import utils.config_util as config_util
from utils.prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
from utils.config_util import RootConfig
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import shutil
from custom_datasets.imagepair import ImageSet
import yaml
from transformers import CLIPTextModel
# from utils.metrics import Clip_metric, CSD_CLIP,LPIPS_metric
import warnings
# from inference import get_validation_dataloader, inference, infer_metric
# import copy
warnings.filterwarnings("ignore")
from diffusers import logging as diffusers_logging
import matplotlib.pyplot as plt
diffusers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    args,
    config: RootConfig,
    device: int,
    style_folder: str,
    # folders,
    # scales,
):
    # scales = np.array(scales)
    # folders = np.array(folders)
    # scales_unique = list(scales)

    painting_folder = os.path.join(style_folder, "paintings")
    caption_folder = os.path.join(style_folder, "captions_long")

    metadata = {
        # "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)
    # save args
    with open(save_path / "train_args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)


    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae

    noise_scheduler = model_util.create_noise_scheduler(
        config.train.noise_scheduler,
        prediction_type="v_prediction" if config.pretrained_model.v_pred else "epsilon",
    )
    print(f"noise_scheduler: {noise_scheduler}")

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    # unet_copy = copy.deepcopy(unet).to("cpu")
    
    vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    network = network.train()
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    #optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    flush()

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.resolution, scale=args.transform_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)),
        ]
    )
    def preprocess(example):
        if "image" in example:
            example["image"] = train_transforms(example["image"])
        if "caption" in example:
            if isinstance(text_encoder, CLIPTextModel):
                # trunc prompts for clip encoder
                example["caption"] = [p.split(".")[0]+"." for p in example["caption"]]
        return example

    dataset = ImageSet(painting_folder, transform=preprocess, keep_in_mem=True, caption=caption_folder)
    if args.train_num is not None and args.train_num > 0:
        dataset.limit_num(args.train_num)

    print(f"Dataset size: {len(dataset)}")
    collate_fn = dataset.collate_fn
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    i=0
    pbar = tqdm(range(config.train.iterations))
    all_losses=[]
    recon_losses = []
    content_losses = []
    while i < config.train.iterations:
    # for i in pbar:
        for batch_i,batch in enumerate(train_dataloader):
            bsz = len(batch['images'])
            network = network.train()
            with torch.no_grad():
                noise_scheduler.set_timesteps(
                    config.train.max_denoising_steps, device=device
                )

                optimizer.zero_grad()


                timesteps_to = torch.randint(
                    1, config.train.max_denoising_steps-1, (1,)
    #                 1, 25, (1,)
                ).item()
                current_timestep = noise_scheduler.timesteps[timesteps_to:timesteps_to+1]

                scale_to_look = args.scales
                img = batch['images']
                batch_size = len(img)

            ori_prompt = batch["captions"] if "captions" in batch else ""
            ori_prompt_embed = train_util.encode_prompts(tokenizer, text_encoder, ori_prompt)
            if batch_i ==0 and i == 0:
                print(f"ori_prompt: {ori_prompt}")
            if args.style_label is not None:
                style_prompt = [f"{p[::-1].replace('.', '',1)[::-1].strip()} in the style of {args.style_label}" for p in ori_prompt]
                positve_prompt_embed = train_util.encode_prompts(tokenizer, text_encoder, style_prompt)
                if batch_i ==0 and i == 0:
                    print(f"style_prompt: {style_prompt}")
            else:
                positve_prompt_embed = ori_prompt_embed
            seed = random.randint(0,2*15)

            generator = torch.manual_seed(seed)
            noised_latents_high, high_noise, init_latents = train_util.get_noisy_image(
                img,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            noised_latents_high = noised_latents_high.to(device, dtype=weight_dtype)
            high_noise = high_noise.to(device, dtype=weight_dtype)
            noise_scheduler.set_timesteps(1000)

            # current_timestep = noise_scheduler.timesteps[
            #     int(timesteps_to * 1000 / config.train.max_denoising_steps)
            # ]

            network.set_lora_slider(scale=scale_to_look)
            latent_model_input = noise_scheduler.scale_model_input(noised_latents_high, current_timestep)
            with network:
                target_latents_high = unet(latent_model_input,
                                       current_timestep,
                                       encoder_hidden_states=positve_prompt_embed).sample


            loss = 0
            loss_high = criteria(target_latents_high, high_noise)#.cpu().to(torch.float32))
            loss += loss_high

            recon_losses.append(loss_high.item())
            pbar_description = f"Loss*1k: {loss_high.item()*1000:.4f}"
            ##########################
            # content perservation loss
            if args.preservation_weight!=0:
                timesteps = torch.tensor(np.random.choice(np.arange(noise_scheduler.config.num_train_timesteps), size=bsz), device=device)
                preservation_noised_latent = noise_scheduler.add_noise(init_latents, high_noise, timesteps)
                preservation_noised_latent = preservation_noised_latent.to(device, dtype=weight_dtype)
                # ori model knowledge
                with torch.no_grad():
                    latents_ori = unet(preservation_noised_latent,
                                       timesteps,
                                       encoder_hidden_states=ori_prompt_embed).sample
                with network:
                    target_latents_high_ori_prompt = unet(preservation_noised_latent,
                                                          timesteps,
                                                          encoder_hidden_states=ori_prompt_embed).sample
                content_loss_high = args.preservation_weight * criteria(target_latents_high_ori_prompt, latents_ori)

                loss += content_loss_high
                content_losses.append(content_loss_high.item())
                pbar_description += f", prompt_content_loss*1k: {content_loss_high.item()*1000:.4f}"

            all_losses.append(loss.item())
            loss.backward()

            ## NOTICE NO zero_grad between these steps (accumulating gradients)
            #following guidelines from Ostris (https://github.com/ostris/ai-toolkit)
            del (
                # high_latents,
                # low_latents,
                target_latents_high,
            )

            pbar.set_description(pbar_description)
            optimizer.step()
            lr_scheduler.step()


            flush()

            if (
                i % config.save.per_steps == 0
                and i != 0
                and i != config.train.iterations - 1
            ):
                print("Saving...")
                save_path.mkdir(parents=True, exist_ok=True)
                network.save_weights(
                    save_path / f"{config.save.name}_{i}steps.pt",
                    dtype=save_weight_dtype,
                )

            pbar.update(1)
            i+=1
            if i >= config.train.iterations:
                break

            flush()

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_{i}steps.pt",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        optimizer,
        network,
        tokenizer,
        text_encoder,
    )

    flush()

    # plot losses figures

    plt.plot(all_losses)
    plt.title('All Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'{save_path}/all_losses.png')
    plt.close()

    plt.plot(recon_losses)
    plt.title('Reconstruction Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'{save_path}/recon_losses.png')

    plt.plot(content_losses)
    plt.title('Content Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'{save_path}/content_losses.png')

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    config.train.noise_scheduler = args.noise_scheduler
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.network.training_method = args.training_method
    config.save.path = args.save_path
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'
    config.save.per_steps = args.save_per_steps
    config.train.lr = args.lr
    config.train.iterations = args.iterations
    config.pretrained_model.ckpt_path = args.pretrained_model_name_or_path

    device = torch.device(f"cuda:{args.device}")

    # save config file
    save_path = Path(config.save.path)
    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_file, save_path / "config.yaml")

    code_save_path = save_path / "code"
    code_save_path.mkdir(parents=True, exist_ok=True)
    current_file = os.path.basename(__file__)
    shutil.copy(current_file, code_save_path / current_file)
    os.makedirs(code_save_path / "utils", exist_ok=True)
    shutil.copy("utils/metrics.py", code_save_path / "utils/metrics.py")

    print(args.style_folder, args.scales)
    # if len(scales) != len(folders):
    #     raise Exception('the number of folders need to match the number of scales')

    train(args=args, config=config, device=device, style_folder = args.style_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default = 'data/config.yaml',
        help="Config file for training.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="LoRA weight.",
    )
    
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=1,
    )
    
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="adapter",
        help="name",
    )
    
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle",
    )
    
    parser.add_argument(
        "--style_folder",
        type=str,
        required=True,
        help="The folder to check",
    )

    parser.add_argument(
        "--scales",
        type=int,
        required=False,
        default = 1,
        help="scales for different attribute-scaled images",
    )
    parser.add_argument("--save_path", type=str, default = "lora_models", help="save path")
    parser.add_argument("--lr", type=float, required=False, default=2e-4)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="rhfeiyang/art-free-diffusion-v1")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--training_method", type=str, default="all_up", help="all_up ,full, noxattn, down_2_attn")
    parser.add_argument("--train_batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_per_steps", type=int, default=200)
    parser.add_argument("--preservation_weight", type=float, default=50)
    parser.add_argument("--infer_mode", default="sks_art",  help="inference mode") #choices=["style", "ori", "sks", "artist", "sks_art","Peter"],
    parser.add_argument("--noise_scheduler", default="ddim", choices=["ddim", "ddpm"], help="noise scheduler")
    parser.add_argument("--train_num", default=None, type=int, help="train sample number")
    parser.add_argument("--transform_scale", default=[0.9, 1.0], type=float, nargs=2, help="transform scale")
    args = parser.parse_args()


    style_root = os.path.dirname(args.style_folder)
    if args.infer_mode == "style":
        with open(os.path.join(style_root, "style_label.txt"), 'r') as f:
            args.style_label = f.readlines()[0].strip()
    elif args.infer_mode == "artist":
        with open(os.path.join(style_root, "style_label.txt"), 'r') as f:
            args.style_label = f.readlines()[0].strip()
            args.style_label = args.style_label.split(",")[0].strip()
    elif args.infer_mode == "ori":
        args.style_label = None
    else:
        args.style_label = args.infer_mode.replace("_", " ")

    print(args.__dict__)

    main(args)
