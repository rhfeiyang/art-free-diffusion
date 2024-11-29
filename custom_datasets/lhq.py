# Authors: Hui Ren (rhfeiyang.github.io)
import os
import pickle
import random
import shutil
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LhqDataset(Dataset):
    def __init__(self, image_folder_path:str, caption_folder_path:str, id_file:str = "clip_dissection/lhq/idx/subsample_100.pickle", transforms: transforms = None,
                 get_img=True,
                 get_cap=True,):

        if isinstance(id_file, list):
            self.ids = id_file
        elif isinstance(id_file, str):
            with open(id_file, 'rb') as f:
                print(f"Loading ids from {id_file}", flush=True)
                self.ids = pickle.load(f)
                print(f"Loaded ids from {id_file}", flush=True)
        self.image_folder_path = image_folder_path
        self.caption_folder_path = caption_folder_path
        self.transforms = transforms
        self.column_names = ["image", "text"]
        self.get_img = get_img
        self.get_cap = get_cap

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        id = self.ids[index]
        ret={"id":id}
        if self.get_img:
            image = self._load_image(id)
            ret["image"]=image
        if self.get_cap:
            target = self._load_caption(id)
            ret["caption"]=[target]
        if self.transforms is not None:
            ret = self.transforms(ret)
        return ret

    def _load_image(self, id: int):
        image_path = f"{self.image_folder_path}/{id}.jpg"
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
        return img

    def _load_caption(self, id: int):
        caption_path = f"{self.caption_folder_path}/{id}.txt"
        with open(caption_path, 'r') as f:
            caption_file = f.read()
        caption = []
        for line in caption_file.split("\n"):
            line = line.strip()
            if len(line) > 0:
                caption.append(line)
        return caption

    def subsample(self, n: int = 10000):
        if n is None or n == -1:
            return self
        ori_len = len(self)
        assert n <= ori_len
        # equal interval subsample
        ids = self.ids[::ori_len // n][:n]
        self.ids = ids
        print(f"LHQ dataset subsampled from {ori_len} to {len(self)}")
        return self

    def with_transform(self, transform):
        self.transforms = transform
        return self


def generate_idx(data_folder = "/data/vision/torralba/clip_dissection/huiren/lhq/lhq_1024_jpg/lhq_1024_jpg/", save_path = "/data/vision/torralba/clip_dissection/huiren/lhq/idx/all_ids.pickle"):
    all_ids = os.listdir(data_folder)
    all_ids = [i.split(".")[0] for i in all_ids if i.endswith(".jpg") or i.endswith(".png")]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pickle.dump(all_ids, open(f"{save_path}", "wb"))
    print("all_ids generated")
    return all_ids

def random_sample(all_ids, sample_num = 110, save_root = "/data/vision/torralba/clip_dissection/huiren/lhq/subsample"):
    chosen_id = random.sample(all_ids, sample_num)
    save_dir = f"{save_root}/{sample_num}"
    os.makedirs(save_dir, exist_ok=True)
    for id in chosen_id:
        img_path = f"/data/vision/torralba/clip_dissection/huiren/lhq/lhq_1024_jpg/lhq_1024_jpg/{id}.jpg"
        shutil.copy(img_path, save_dir)

    return chosen_id

if __name__ == "__main__":
    # all_ids = generate_idx()
    # with open("/data/vision/torralba/clip_dissection/huiren/lhq/idx/all_ids.pickle", "rb") as f:
    #     all_ids = pickle.load(f)
    # # random_sample(all_ids, 1)
    #
    # # generate_idx(data_folder="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/lhq/subsample/100",
    # #              save_path="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/lhq/idx/subsample_100.pickle")
    #
    # # lhq 500
    # with open("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/lhq/idx/subsample_100.pickle", "rb") as f:
    #     lhq_100_idx = pickle.load(f)
    #
    # extra_idx = set(all_ids) - set(lhq_100_idx)
    # add_idx = random.sample(extra_idx, 400)
    # lhq_500_idx = lhq_100_idx + add_idx
    # with open("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/lhq/idx/subsample_500.pickle", "wb") as f:
    #     pickle.dump(lhq_500_idx, f)
    # save_dir = "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/lhq/subsample/500"
    # os.makedirs(save_dir, exist_ok=True)
    # for id in lhq_500_idx:
    #     img_path = f"/data/vision/torralba/clip_dissection/huiren/lhq/lhq_1024_jpg/lhq_1024_jpg/{id}.jpg"
    #     # softlink
    #     os.symlink(img_path, os.path.join(save_dir, f"{id}.jpg"))

    # lhq9
    all_ids = generate_idx(data_folder="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/lhq/subsample/9",
                           save_path="/data/vision/torralba/clip_dissection/huiren/lhq/idx/subsample_9.pickle")
    print(all_ids)


    
