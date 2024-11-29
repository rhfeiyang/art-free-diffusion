# Authors: Hui Ren (rhfeiyang.github.io)

import os

import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from collections import OrderedDict
from transformers import BatchFeature
import clip
import copy
import lpips
from transformers import ViTImageProcessor, ViTModel

## CSD_CLIP
def convert_weights_float(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


## taken from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py
class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim,
            dropout=0
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)

class Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_preprocess = None

    def load_image(self, image_path):
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert("RGB")
        return image

    def load_image_path(self, image_path):
        if isinstance(image_path, str):
            # should be a image folder path
            images_file = os.listdir(image_path)
            images = [os.path.join(image_path, image) for image in images_file if
                      image.endswith(".jpg") or image.endswith(".png")]
        if isinstance(image_path[0], str):
            images = [self.load_image(image) for image in image_path]
        elif isinstance(image_path[0], np.ndarray):
            images = [Image.fromarray(image) for image in image_path]
        elif isinstance(image_path[0], Image.Image):
            images = image_path
        else:
            raise Exception("Invalid input")
        return images

    def preprocess_image(self, image, **kwargs):
        if (isinstance(image, str) and os.path.isdir(image)) or (isinstance(image, list) and (isinstance(image[0], Image.Image) or isinstance(image[0], np.ndarray) or os.path.isfile(image[0]))):
            input_data = self.load_image_path(image)
            input_data = [self.image_preprocess(image, **kwargs) for image in input_data]
            input_data = torch.stack(input_data)
        elif os.path.isfile(image):
            input_data = self.load_image(image)
            input_data = self.image_preprocess(input_data, **kwargs)
            input_data = input_data.unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            raise Exception("Unsupported input")
        return input_data

class Clip_Basic_Metric(Metric):
    def __init__(self):
        super().__init__()
        self.tensor_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.rescale
            transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.image_preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class Clip_metric(Clip_Basic_Metric):

    @torch.no_grad()
    def __init__(self, target_style_prompt: str=None, clip_model_name="openai/clip-vit-large-patch14", device="cuda",
                 bath_size=8, alpha=0.5):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.model = (CLIPModel.from_pretrained(clip_model_name)).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        # self.style_class_features = self.get_text_features(self.styles).cpu()
        self.style_class_features=[]
        # self.noise_prompt_features = self.get_text_features("Noise")
        self.model.eval()
        self.batch_size = bath_size
        if target_style_prompt is not None:
            self.ref_style_features = self.get_text_features(target_style_prompt)
        else:
            self.ref_style_features = None

        self.ref_image_style_prototype = None

    def get_text_features(self, text):
        prompt_encoding = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        prompt_features = self.model.get_text_features(**prompt_encoding).to(self.device)
        prompt_features = F.normalize(prompt_features, p=2, dim=-1)
        return prompt_features

    def get_image_features(self, images):
        # if isinstance(image, torch.Tensor):
        #     self.tensor_transform(image)
        # else:
        #     image_features = self.image_processor(image, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
        images = self.load_image_path(images)
        if isinstance(images, torch.Tensor):
            images = self.tensor_preprocess(images)
            data = {"pixel_values": images}
            image_features = BatchFeature(data=data, tensor_type="pt")
        else:
            image_features = self.image_processor(images, return_tensors="pt", padding=True).to(self.device,
                                                                                                non_blocking=True)

        image_features = self.model.get_image_features(**image_features).to(self.device)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def img_text_similarity(self, image_features, text=None):
        if text is not None:
            prompt_feature = self.get_text_features(text)
            if isinstance(text, str):
                prompt_feature = prompt_feature.repeat(len(image_features), 1)
        else:
            prompt_feature = self.ref_style_features

        similarity_each = torch.einsum("nc, nc -> n", image_features, prompt_feature)
        return similarity_each

    def forward(self, output_imgs, prompt=None):
        image_features = self.get_image_features(output_imgs)
        # print(image_features)
        style_score = self.img_text_similarity(image_features.mean(dim=0, keepdim=True))
        if prompt is not None:
            content_score = self.img_text_similarity(image_features, prompt)

            score = self.alpha * style_score + (1 - self.alpha) * content_score
            return {"score": score, "style_score": style_score, "content_score": content_score}
        else:
            return {"style_score": style_score}

    def content_score(self, output_imgs, prompt):
        self.to(self.device)
        image_features = self.get_image_features(output_imgs)
        content_score_details = self.img_text_similarity(image_features, prompt)
        self.to("cpu")
        return {"CLIP_content_score": content_score_details.mean().cpu(), "CLIP_content_score_details": content_score_details.cpu()}


class CSD_CLIP(Clip_Basic_Metric):
    """backbone + projection head"""
    def __init__(self, name='vit_large',content_proj_head='default', ckpt_path = "data/weights/CSD-checkpoint.pth", device="cuda",
                 alpha=0.5, **kwargs):
        super(CSD_CLIP, self).__init__()
        self.alpha = alpha
        self.content_proj_head = content_proj_head
        self.device = device
        if name == 'vit_large':
            clipmodel, _ = clip.load("ViT-L/14")
            self.backbone = clipmodel.visual
            self.embedding_dim = 1024
        elif name == 'vit_base':
            clipmodel, _ = clip.load("ViT-B/16")
            self.backbone = clipmodel.visual
            self.embedding_dim = 768
            self.feat_dim = 512
        else:
            raise Exception('This model is not implemented')

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        if content_proj_head == 'custom':
            self.last_layer_content = ProjectionHead(self.embedding_dim,self.feat_dim)
            self.last_layer_content.apply(init_weights)

        else:
            self.last_layer_content = copy.deepcopy(self.backbone.proj)

        self.backbone.proj = None
        self.backbone.requires_grad_(False)
        self.last_layer_style.requires_grad_(False)
        self.last_layer_content.requires_grad_(False)
        self.backbone.eval()

        if ckpt_path is not None:
            self.load_ckpt(ckpt_path)
        self.to("cpu")

    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"=> loaded CSD_CLIP checkpoint with msg {msg}")

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def get_image_features(self, input_data, get_style=True,get_content=False,feature_alpha=None):
        if isinstance(input_data, torch.Tensor):
            input_data = self.tensor_preprocess(input_data)
        elif (isinstance(input_data, str) and os.path.isdir(input_data)) or (isinstance(input_data, list) and (isinstance(input_data[0], Image.Image) or isinstance(input_data[0], np.ndarray) or os.path.isfile(input_data[0]))):
            input_data = self.load_image_path(input_data)
            input_data = [self.image_preprocess(image) for image in input_data]
            input_data = torch.stack(input_data)
        elif os.path.isfile(input_data):
            input_data = self.load_image(input_data)
            input_data = self.image_preprocess(input_data)
            input_data = input_data.unsqueeze(0)
        input_data = input_data.to(self.device)
        style_output = None

        feature = self.backbone(input_data)
        if get_style:
            style_output = feature @ self.last_layer_style
            # style_output = style_output.mean(dim=0)
            style_output = nn.functional.normalize(style_output, dim=-1, p=2)

        content_output=None
        if get_content:
            if feature_alpha is not None:
                reverse_feature = ReverseLayerF.apply(feature, feature_alpha)
            else:
                reverse_feature = feature
            # if alpha is not None:
            if self.content_proj_head == 'custom':
                content_output =  self.last_layer_content(reverse_feature)
            else:
                content_output = reverse_feature @ self.last_layer_content
            content_output = nn.functional.normalize(content_output, dim=-1, p=2)

        return feature, content_output, style_output


    @torch.no_grad()
    def define_ref_image_style_prototype(self, ref_image_path: str):
        self.to(self.device)
        _, _, self.ref_style_feature = self.get_image_features(ref_image_path)
        self.to("cpu")
        # self.ref_style_feature = self.ref_style_feature.mean(dim=0)
    @torch.no_grad()
    def forward(self, styled_data):
        self.to(self.device)
        # get_content_feature = original_data is not None
        _, content_output, style_output = self.get_image_features(styled_data, get_content=False)
        style_similarities = style_output @ self.ref_style_feature.T
        mean_style_similarities = style_similarities.mean(dim=-1)
        mean_style_similarity = mean_style_similarities.mean()

        max_style_similarities_v, max_style_similarities_id = style_similarities.max(dim=-1)
        max_style_similarity = max_style_similarities_v.mean()


        self.to("cpu")
        return {"CSD_similarity_mean": mean_style_similarity, "CSD_similarity_max": max_style_similarity, "CSD_similarity_mean_details": mean_style_similarities,
                "CSD_similarity_max_v_details": max_style_similarities_v, "CSD_similarity_max_id_details": max_style_similarities_id}

    def get_style_loss(self, styled_data):
        _, _, style_output = self.get_image_features(styled_data, get_style=True, get_content=False)
        style_similarity = (style_output @ self.ref_style_feature).mean()
        loss = 1 - style_similarity
        return loss.mean()

class LPIPS_metric(Metric):
    def __init__(self, type="vgg", device="cuda"):
        super(LPIPS_metric, self).__init__()
        self.lpips = lpips.LPIPS(net=type)
        self.device = device
        self.image_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.to("cpu")

    @torch.no_grad()
    def forward(self, img1, img2):
        self.to(self.device)
        differences = []
        for i in range(0, len(img1), 50):
            img1_batch = img1[i:i+50]
            img2_batch = img2[i:i+50]
            img1_batch = self.preprocess_image(img1_batch).to(self.device)
            img2_batch = self.preprocess_image(img2_batch).to(self.device)
            differences.append(self.lpips(img1_batch, img2_batch).squeeze())
        differences = torch.cat(differences)
        difference = differences.mean()
        # similarity = 1 - difference
        self.to("cpu")
        return {"LPIPS_content_difference": difference,  "LPIPS_content_difference_details": differences}

class Vit_metric(Metric):
    def __init__(self, device="cuda"):
        super(Vit_metric, self).__init__()
        self.device = device
        self.model = ViTModel.from_pretrained('facebook/dino-vitb8').eval()
        self.image_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
        self.to("cpu")
    def get_image_features(self, images):
        # if isinstance(image, torch.Tensor):
        #     self.tensor_transform(image)
        # else:
        #     image_features = self.image_processor(image, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
        images = self.load_image_path(images)
        batch_size = 20
        all_image_features = []
        for i in range(0, len(images), batch_size):
            image_batch = images[i:i+batch_size]
            if isinstance(image_batch, torch.Tensor):
                image_batch = self.tensor_preprocess(image_batch)
                data = {"pixel_values": image_batch}
                image_processed = BatchFeature(data=data, tensor_type="pt")
            else:
                image_processed = self.image_processor(image_batch, return_tensors="pt").to(self.device)
            image_features = self.model(**image_processed).last_hidden_state.flatten(start_dim=1)
            image_features = F.normalize(image_features, p=2, dim=-1)
            all_image_features.append(image_features)
        all_image_features = torch.cat(all_image_features)
        return all_image_features

    @torch.no_grad()
    def content_metric(self, img1, img2):
        self.to(self.device)
        if not(isinstance(img1, torch.Tensor) and len(img1.shape) == 2):
            img1 = self.get_image_features(img1)
        if not(isinstance(img2, torch.Tensor) and len(img2.shape) == 2):
            img2 = self.get_image_features(img2)
        similarities = torch.einsum("nc, nc -> n", img1, img2)
        similarity = similarities.mean()
        # self.to("cpu")
        return {"Vit_content_similarity": similarity, "Vit_content_similarity_details": similarities}

    # style
    @torch.no_grad()
    def define_ref_image_style_prototype(self, ref_image_path: str):
        self.to(self.device)
        self.ref_style_feature = self.get_image_features(ref_image_path)
        self.to("cpu")
    @torch.no_grad()
    def style_metric(self, styled_data):
        self.to(self.device)
        if isinstance(styled_data, torch.Tensor) and len(styled_data.shape) == 2:
            style_output = styled_data
        else:
            style_output = self.get_image_features(styled_data)
        style_similarities = style_output @ self.ref_style_feature.T
        mean_style_similarities = style_similarities.mean(dim=-1)
        mean_style_similarity = mean_style_similarities.mean()

        max_style_similarities_v, max_style_similarities_id = style_similarities.max(dim=-1)
        max_style_similarity = max_style_similarities_v.mean()

        # self.to("cpu")
        return {"Vit_style_similarity_mean": mean_style_similarity, "Vit_style_similarity_max": max_style_similarity, "Vit_style_similarity_mean_details": mean_style_similarities,
                "Vit_style_similarity_max_v_details": max_style_similarities_v, "Vit_style_similarity_max_id_details": max_style_similarities_id}
    @torch.no_grad()
    def forward(self, styled_data, original_data=None):
        self.to(self.device)
        styled_features = self.get_image_features(styled_data)
        ret ={}
        if original_data is not None:
            content_metric = self.content_metric(styled_features, original_data)
            ret["Vit_content"] = content_metric
        style_metric = self.style_metric(styled_features)
        ret["Vit_style"] = style_metric
        self.to("cpu")
        return ret



class StyleContentMetric(nn.Module):
    def __init__(self, style_ref_image_folder, device="cuda"):
        super(StyleContentMetric, self).__init__()
        self.device = device
        self.clip_style_metric = CSD_CLIP(device=device)
        self.ref_image_file = os.listdir(style_ref_image_folder)
        self.ref_image_file = [i for i in self.ref_image_file if i.endswith(".jpg") or i.endswith(".png")]
        self.ref_image_file.sort()
        self.ref_image_file = np.array(self.ref_image_file)
        ref_image_file_path = [os.path.join(style_ref_image_folder, i) for i in self.ref_image_file]

        self.clip_style_metric.define_ref_image_style_prototype(ref_image_file_path)
        self.vit_metric = Vit_metric(device=device)
        self.vit_metric.define_ref_image_style_prototype(ref_image_file_path)
        self.lpips_metric = LPIPS_metric(device=device)

        self.clip_content_metric = Clip_metric(alpha=0, target_style_prompt=None)

        self.to("cpu")

    def forward(self, styled_data, original_data=None, content_caption=None):
        ret ={}
        csd_score = self.clip_style_metric(styled_data)
        csd_score["max_query"] = self.ref_image_file[csd_score["CSD_similarity_max_id_details"].cpu()].tolist()
        torch.cuda.empty_cache()
        ret["Style_CSD"] = csd_score
        vit_score = self.vit_metric(styled_data, original_data)
        torch.cuda.empty_cache()
        vit_style = vit_score["Vit_style"]
        vit_style["max_query"] = self.ref_image_file[vit_style["Vit_style_similarity_max_id_details"].cpu()].tolist()
        ret["Style_VIT"] = vit_style

        if original_data is not None:
            vit_content = vit_score["Vit_content"]
            ret["Content_VIT"] = vit_content
            lpips_score = self.lpips_metric(styled_data, original_data)
            torch.cuda.empty_cache()
            ret["Content_LPIPS"] = lpips_score

        if content_caption is not None:
            clip_content = self.clip_content_metric.content_score(styled_data, content_caption)
            ret["Content_CLIP"] = clip_content
            torch.cuda.empty_cache()

        for type_key, type_value in ret.items():
            for key, value in type_value.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        ret[type_key][key] = round(value.item(), 4)
                    else:
                        ret[type_key][key] = value.tolist()
                        ret[type_key][key] = [round(v, 4) for v in ret[type_key][key]]

        self.to("cpu")
        ret["ref_image_file"] = self.ref_image_file.tolist()
        return ret


if __name__ == "__main__":
    with torch.no_grad():
        metric = StyleContentMetric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/Art_styles/camille-pissarro/impressionism/split_5/paintings")
        score = metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/converted_photo/500",
                       "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/paintings")
        print(score)



        lpips = LPIPS_metric()
        score = lpips("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/paintings",
                      "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/converted_photo/500")

        print("lpips", score)


        clip_metric = CSD_CLIP()
        clip_metric.define_ref_image_style_prototype(
            "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset1/paintings")

        score = clip_metric(
            "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/converted_photo/500")
        print("subset3-subset3_sd14_converted", score)

        score = clip_metric(
            "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/clip_filtered_remain_500")
        print("subset3-photo", score)



        score = clip_metric(
            "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset1/paintings")
        print("subset3-subset1", score)

        score = clip_metric(
            "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/andy-warhol/pop_art/subset1/paintings")
        print("subset3-andy", score)
        # score = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/paintings", "A painting")

        # print("subset3",score)
        # score_subset2 = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset2/paintings")
        # print("subset2",score_subset2)
        # score_subset3 = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/paintings")
        # print("subset3",score_subset3)
        #
        # score_subset3_converted = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/converted_photo/500")
        # print("subset3-subset3_sd14_converted" , score_subset3_converted)
        #
        # score_subset3_coco_converted = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/coco_converted_photo/500")
        # print("subset3-subset3_coco_converted" , score_subset3_coco_converted)
        #
        # clip_metric = Clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/sketch_500")
        # score = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/clip_filtered_remain_500")
        # print("photo500_1-sketch" ,score)
        #
        # clip_metric = Clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/clip_filtered_remain_500")
        # score = clip_metric("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/clip_filtered_remain_500_new")
        # print("photo500_1-photo500_2" ,score)
        # from custom_datasets.imagepair import ImageSet
        # import matplotlib.pyplot as plt
        # dataset = ImageSet(folder = "/data/vision/torralba/scratch/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/paintings",
        #                    caption_path="/data/vision/torralba/scratch/huiren/code/diffusion/stable_diffusion/custom_datasets/wikiart/data/gustav-klimt_Art_Nouveau/subset3/captions",
        #                     keep_in_mem=False)
        # for sample in dataset:
        #     score = clip_metric.content_score(sample["image"], sample["caption"][0])
        #     plt.imshow(sample["image"])
        #     plt.title(f"score: {round(score.item(),2)}\n prompt: {sample['caption'][0]}")
        #     plt.show()
