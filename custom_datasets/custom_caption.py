# Authors: Hui Ren (rhfeiyang.github.io)
import torch
import pandas as pd
import numpy as np
import os
from PIL import Image

class Caption_set(torch.utils.data.Dataset):

    style_set_names=[
        "andre-derain_subset1",
        "andy_subset1",
        "camille-corot_subset1",
        "gerhard-richter_subset1",
        "henri-matisse_subset1",
        "katsushika-hokusai_subset1",
        "klimt_subset3",
        "monet_subset2",
        "picasso_subset1",
        "van_gogh_subset1",
    ]
    style_set_map={f"{name}":f"/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/Style_captions/{name}/style_captions.csv" for name in style_set_names}

    def __init__(self, prompts_path=None, set_name=None, transform=None):
        assert prompts_path is not None or set_name is not None, "Either prompts_path or set_name should be provided"
        if prompts_path is None:
            prompts_path = self.style_set_map[set_name]

        self.prompts = pd.read_csv(prompts_path, delimiter=';')
        self.transform = transform
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        ret={}
        ret["id"] = idx
        info = self.prompts.iloc[idx]
        ret.update(info)
        for k,v in ret.items():
            if isinstance(v,np.int64):
                ret[k] = int(v)
        ret["caption"] = [ret["caption"]]
        if self.transform:
            ret = self.transform(ret)
        return ret

    def with_transform(self, transform):
        self.transform = transform
        return self


class HRS_caption(Caption_set):
    def __init__(self, prompts_path="/vision-nfs/torralba/projects/jomat/hui/stable_diffusion/clip_dissection/Style_captions/andre-derain_subset1/style_captions.csv", transform=None, delimiter=','):
        self.prompts = pd.read_csv(prompts_path, delimiter=delimiter)
        self.transform = transform
        self.caption_key = "original_prompts"

    def __getitem__(self, idx):
        ret={}
        ret["id"] = idx
        info = self.prompts.iloc[idx]
        ret["caption"] = [info[self.caption_key]]
        ret["seed"] = idx
        if self.transform:
            ret = self.transform(ret)
        return ret

class Laion_pop(torch.utils.data.Dataset):
    def __init__(self, anno_file="/vision-nfs/torralba/projects/jomat/hui/stable_diffusion/custom_datasets/laion_pop500.csv",image_root="/vision-nfs/torralba/scratch/jomat/sam_dataset/laion_pop",transform=None):
        self.transform = transform
        self.info = pd.read_csv(anno_file, delimiter=";")
        self.caption_key = "caption"
        self.image_root = image_root
        self.get_img=True
        self.get_caption=True
    def __len__(self):
        return len(self.info)

    # def subsample(self, num:int):
    #     self.data = self.data.select(range(num))
    #     return self

    def load_image(self, key):
        image_path = os.path.join(self.image_root, f"{key:09}.jpg")
        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")
        return image

    def __getitem__(self, idx):
        info = self.info.iloc[idx]
        ret = {}
        key = info["key"]
        ret["id"] = key
        if self.get_caption:
            ret["caption"] = [info[self.caption_key]]
        ret["seed"] = int(key)
        if self.get_img:
            ret["image"] = self.load_image(key)

        if self.transform:
            ret = self.transform(ret)
        return ret

    def with_transform(self, transform):
        self.transform = transform
        return self

    def subset(self, ids:list):
        self.info = self.info[self.info["key"].isin(ids)]
        return self

if __name__ == "__main__":
    dataset = Caption_set("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/Style_captions/andre-derain_subset1/style_captions.csv")
    dataset[0]
