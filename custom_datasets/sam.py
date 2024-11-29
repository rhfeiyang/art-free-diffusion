# Authors: Hui Ren (rhfeiyang.github.io)
import os.path
import sys
from typing import Any, Callable, List, Optional, Tuple

import tqdm
from PIL import Image

from torch.utils.data import Dataset
import pickle
from torchvision import transforms
# import torch
# import torchvision
# import re


class SamDataset(Dataset):
    def __init__(self, image_folder_path:str, caption_folder_path:str, id_file:str = "data/sam/clip_filtered_ids.pickle",id_dict_file:str =None , transforms: Optional[Callable] = None,
                 resolution=None,
                 get_img=True,
                 get_cap=True,):
        if id_dict_file is not None:
            with open(id_dict_file, 'rb') as f:
                print(f"Loading id_dict from {id_dict_file}", flush=True)
                self.id_dict = pickle.load(f)
                print(f"Loaded id_dict from {id_dict_file}", flush=True)
        else:
            self.id_dict = None
        if isinstance(id_file, list):
            self.ids = id_file
        elif isinstance(id_file, str):
            with open(id_file, 'rb') as f:
                print(f"Loading ids from {id_file}", flush=True)
                self.ids = pickle.load(f)
                print(f"Loaded ids from {id_file}", flush=True)
        self.resolution = resolution
        self.ori_image_folder_path = image_folder_path
        if self.resolution is not None:
            if os.path.exists("/var/jomat/datasets/"):
                # self.image_folder_path = f"/var/jomat/datasets/SAM_{resolution}"
                self.image_folder_path = f"{image_folder_path}_{resolution}"
            else:
                self.image_folder_path = f"{image_folder_path}_{resolution}"
            os.makedirs(self.image_folder_path, exist_ok=True)
        else:
            self.image_folder_path = image_folder_path
        self.caption_folder_path = caption_folder_path
        self.transforms = transforms
        self.column_names = ["image", "text"]
        self.get_img = get_img
        self.get_cap = get_cap

    def __len__(self):
        # return 100
        return len(self.ids)

    def __getitem__(self, index: int):
        id = self.ids[index]
        ret={"id":id}
        try:
            # if index == 1:
            #     raise Exception("test")
            if self.get_img:
                image = self._load_image(id)
                ret["image"]=image
            if self.get_cap:
                target = self._load_caption(id)
                ret["text"] = [target]
            if self.transforms is not None:
                ret = self.transforms(ret)
            return ret
        except Exception as e:
            raise e
            print(f"Error loading image and caption for id {id}, error: {e}, redirecting to index 0", flush=True)
            ret = self[0]
            return ret

    def define_resolution(self, resolution: int):
        self.resolution = resolution
        if os.path.exists("/var/jomat/datasets/"):
            self.image_folder_path = f"/var/jomat/datasets/SAM_{resolution}"
            # self.image_folder_path = f"{self.ori_image_folder_path}_{resolution}"
        else:
            self.image_folder_path = f"{self.ori_image_folder_path}_{resolution}"
        print(f"SamDataset resolution defined to {resolution}, new image folder path: {self.image_folder_path}")
    def _load_image(self, id: int) -> Image.Image:
        if self.id_dict is not None:
            subfolder = self.id_dict[id]
            image_path = f"{self.image_folder_path}/{subfolder}/sa_{id}.jpg"
        else:
            image_path = f"{self.image_folder_path}/sa_{id}.jpg"

        try:
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            # return img
        except:
            # load original image
            if self.id_dict is not None:
                subfolder = self.id_dict[id]
                ori_image_path = f"{self.ori_image_folder_path}/{subfolder}/sa_{id}.jpg"
            else:
                ori_image_path = f"{self.ori_image_folder_path}/sa_{id}.jpg"
            assert os.path.exists(ori_image_path)
            with open(ori_image_path, 'rb') as f:
                img = Image.open(f).convert("RGB")
            # resize image keep aspect ratio
            if self.resolution is not None:
                img = transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC)(img)
            # write image
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            img.save(image_path)

        return img

    
    def _load_caption(self, id: int):
        caption_path = f"{self.caption_folder_path}/sa_{id}.txt"
        if not os.path.exists(caption_path):
            return None
        try:
            with open(caption_path, 'r', encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise e
            print(f"Error reading caption file {caption_path}, error: {e}")
            return None
        sentences = content.split('.')
        # remove empty sentences and sentences with "black and white"(too many false prediction)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip() and "black and white" not in sentence]
        # join sentence
        sentences = ". ".join(sentences)
        if len(sentences) > 0 and sentences[-1] != '.':
            sentences += '.'

        return sentences
    
    def with_transform(self, transform):
        self.transforms = transform
        return self

    def subsample(self, n: int = 10000):
        if n is None or n == -1:
            return self
        ori_len = len(self)
        assert n <= ori_len
        # equal interval subsample
        ids = self.ids[::ori_len // n][:n]
        self.ids = ids
        print(f"SAM dataset subsampled from {ori_len} to {len(self)}")
        return self


if __name__ == "__main__":
    # sam_filt(caption_filt=False, clip_filt=False, clip_logit=True)
    from custom_datasets.sam_caption.mypath import MyPath
    dataset = SamDataset(image_folder_path=MyPath.db_root_dir("sam_images"), caption_folder_path=MyPath.db_root_dir("sam_captions"), id_file=MyPath.db_root_dir("sam_whole_filtered_ids_train"), id_dict_file=MyPath.db_root_dir("sam_id_dict"))
    dataset.get_img = False
    for i in tqdm.tqdm(dataset):
        a=i['text']
