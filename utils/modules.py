# Authors: Hui Ren (rhfeiyang.github.io)
import os
import torch
from torchvision import transforms
from transformers.utils import ContextManagers

from diffusers import AutoencoderKL, PNDMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, BertModel, BertTokenizer
from accelerate.state import AcceleratorState
import accelerate
from diffusers.training_utils import EMAModel
def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

def get_diffusion_modules(unet_config_path = "CompVis/stable-diffusion-v1-4", unet_pretrained=False, vae_path="CompVis/stable-diffusion-v1-4",
                          text_encoder_path="CompVis/stable-diffusion-v1-4",
                          scheduler_path= "CompVis/stable-diffusion-v1-4",
                          revision = None, variant = None, use_ema=False,  **kwargs):

    ret={}
    if unet_config_path is not None:
        unet_config = UNet2DConditionModel.load_config(unet_config_path, subfolder="unet")
        if unet_pretrained is True:
            unet = UNet2DConditionModel.from_pretrained(unet_config_path, subfolder="unet", revision=revision, variant=variant)
        else:
            unet = UNet2DConditionModel.from_config(unet_config)
            if unet_pretrained == "ema":
                ema_unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(unet_config_path, "unet_ema"), subfolder="unet", revision=revision, variant=variant
                )
                ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
                # ema_unet = EMAModel.from_pretrained(os.path.join(unet_config_path, "unet_ema"), UNet2DConditionModel)
                ema_unet.copy_to(unet.parameters())
        ret["unet"] = unet

        if use_ema:
            ema_unet = UNet2DConditionModel.from_config(unet_config)
            ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
            ret["ema_unet"] = ema_unet

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if text_encoder_path is not None:
            if "bert" in text_encoder_path:
                text_encoder = BertModel.from_pretrained(text_encoder_path, revision=revision)
                text_encoder.config.use_attention_mask = True
            else:
                text_encoder = CLIPTextModel.from_pretrained(
                    text_encoder_path, subfolder="text_encoder", revision=revision, variant=variant
                )
            ret["text_encoder"] = text_encoder
        if vae_path is not None:
            vae = AutoencoderKL.from_pretrained(
                vae_path, subfolder="vae", revision=revision, variant=variant
            )
            ret["vae"] = vae
    if text_encoder_path is not None:
        if "bert" in text_encoder_path:
            tokenizer = BertTokenizer.from_pretrained(text_encoder_path)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                text_encoder_path, subfolder="tokenizer", revision=revision
            )
        ret["tokenizer"] = tokenizer

    if scheduler_path is not None:
        scheduler = DDPMScheduler.from_pretrained(scheduler_path, subfolder="scheduler")
        ret["noise_scheduler"] = scheduler
    return ret



