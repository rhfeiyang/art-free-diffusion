import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from torchvision.datasets.vision import VisionDataset
import pickle
import csv
import pandas as pd
import torch
import torchvision
import re
# from torchvision.datasets import CocoDetection
# from utils.clip_filter import Clip_filter
from tqdm import tqdm
from .mypath import MyPath

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str ,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            get_img=True,
            get_cap=True
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.column_names = ["image", "text"]
        self.get_img = get_img
        self.get_cap = get_cap

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        with open(os.path.join(self.root, path), 'rb') as f:
            img = Image.open(f).convert("RGB")

        return img

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        ret={"id":id}
        if self.get_img:
            image = self._load_image(id)
            ret["image"] = image
        if self.get_cap:
            target = self._load_target(id)
            ret["caption"] = [target]

        if self.transforms is not None:
            ret = self.transforms(ret)

        return ret

    def subsample(self, n: int = 10000):
        if n is None or n == -1:
            return self
        ori_len = len(self)
        assert n <= ori_len
        # equal interval subsample
        ids = self.ids[::ori_len // n][:n]
        self.ids = ids
        print(f"COCO dataset subsampled from {ori_len} to {len(self)}")
        return self


    def with_transform(self, transform):
        self.transforms = transform
        return self

    def __len__(self) -> int:
        # return 100
        return len(self.ids)


class CocoCaptions(CocoDetection):
    """`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.PILToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]


class CocoCaptions_clip_filtered(CocoCaptions):
    positive_prompt=["painting", "drawing", "graffiti",]
    def __init__(
            self,
            root: str ,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            regenerate: bool = False,
            id_file: Optional[str] = "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/data/coco/coco_clip_filtered_ids.pickle"
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        os.makedirs(os.path.dirname(id_file), exist_ok=True)
        if os.path.exists(id_file) and not regenerate:
            with open(id_file, "rb") as f:
                self.ids = pickle.load(f)
        else:
            self.ids, naive_filtered_num = self.naive_filter()
            self.ids, clip_filtered_num = self.clip_filter(0.7)

            print(f"naive Filtered {naive_filtered_num} images")
            print(f"Clip Filtered {clip_filtered_num} images")

            with open(id_file, "wb") as f:
                pickle.dump(self.ids, f)
                print(f"Filtered ids saved to {id_file}")
        print(f"COCO filtered dataset size: {len(self)}")

    def naive_filter(self, filter_prompt="painting"):
        new_ids = []
        naive_filtered_num = 0
        for id in self.ids:
            target = self._load_target(id)
            filtered = False
            for prompt in target:
                if filter_prompt in prompt.lower():
                    filtered = True
                    naive_filtered_num += 1
                    break
                # if "artwork" in prompt.lower():
                #     pass
            if not filtered:
                new_ids.append(id)
        return new_ids, naive_filtered_num

    # def clip_filter(self, threshold=0.7):
    #
    #     def collate_fn(examples):
    #         # {"image": image, "text": [target], "id":id}
    #         pixel_values = [example["image"] for example in examples]
    #         prompts = [example["text"] for example in examples]
    #         id = [example["id"] for example in examples]
    #         return {"images": pixel_values, "prompts": prompts, "ids": id}
    #
    #
    #     clip_filtered_num = 0
    #     clip_filter = Clip_filter(positive_prompt=self.positive_prompt)
    #     clip_logs={"positive_prompt":clip_filter.positive_prompt, "negative_prompt":clip_filter.negative_prompt,
    #                "ids":torch.Tensor([]),"logits":torch.Tensor([])}
    #     clip_log_file = "data/coco/clip_logs.pth"
    #     new_ids = []
    #     batch_size = 128
    #     dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=10, shuffle=False,
    #                                              collate_fn=collate_fn)
    #     for i, batch in enumerate(tqdm(dataloader)):
    #         images = batch["images"]
    #         filter_result, logits = clip_filter.filter(images, threshold=threshold)
    #         ids = torch.IntTensor(batch["ids"])
    #         clip_logs["ids"] = torch.cat([clip_logs["ids"], ids])
    #         clip_logs["logits"] = torch.cat([clip_logs["logits"], logits])
    #
    #         new_ids.extend(ids[~filter_result].tolist())
    #         clip_filtered_num += filter_result.sum().item()
    #         if i % 50 == 0:
    #             torch.save(clip_logs, clip_log_file)
    #     torch.save(clip_logs, clip_log_file)
    #
    #     return new_ids, clip_filtered_num


class CustomCocoCaptions(CocoCaptions):
    def __init__(self, root: str=MyPath.db_root_dir("coco_val"), annFile: str=MyPath.db_root_dir("coco_caption_val"), custom_file:str="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/jomat-code/filtering/ms_coco_captions_testset100.txt",transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None) -> None:

        super().__init__(root, annFile, transform, target_transform, transforms)
        self.column_names = ["image", "text"]
        self.custom_file = custom_file
        self.load_custom_data(custom_file)
        self.transforms = transforms

    def load_custom_data(self, custom_file):
        self.custom_data = []
        with open(custom_file, "r") as f:
            data = f.readlines()
        head = data[0].strip().split(",")
        self.head = head
        for line in data[1:]:
            sub_data = line.strip().split(",")
            if len(sub_data) > len(head):
                sub_data_new = [sub_data[0]]
                sub_data_new+=[",".join(sub_data[1:-1])]
                sub_data_new.append(sub_data[-1])
                sub_data = sub_data_new
            assert len(sub_data) == len(head)
            self.custom_data.append(sub_data)
        # to pd
        self.custom_data = pd.DataFrame(self.custom_data, columns=head)

    def __len__(self) -> int:
        return len(self.custom_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.custom_data.iloc[index]
        id = int(data["image_id"])
        ret={"id":id}
        if self.get_img:
            image = self._load_image(id)
            ret["image"] = image
        if self.get_cap:
            caption = data["caption"]
            ret["caption"] = [caption]
        ret["seed"] = int(data["random_seed"])

        if self.transforms is not None:
            ret = self.transforms(ret)

        return ret



def get_validation_set():
    coco_instance = CocoDetection(root="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/.datasets/coco_2017/train2017/", annFile="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/.datasets/coco_2017/annotations/instances_train2017.json")
    discard_cat_id = coco_instance.coco.getCatIds(supNms=["person", "animal"])
    discard_img_id = []
    for cat_id in discard_cat_id:
        discard_img_id += coco_instance.coco.catToImgs[cat_id]

    coco_clip_filtered = CocoCaptions_clip_filtered(root=MyPath.db_root_dir("coco_train"), annFile=MyPath.db_root_dir("coco_caption_train"),
                                regenerate=False)
    coco_clip_filtered_ids = coco_clip_filtered.ids
    new_ids = set(coco_clip_filtered_ids) - set(discard_img_id)
    new_ids = list(new_ids)
    new_ids = random.sample(new_ids, 100)
    with open("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/data/coco/coco_clip_filtered_subset100.pickle", "wb") as f:
        pickle.dump(new_ids, f)

if __name__ == "__main__":
    from mypath import MyPath
    import random
    # get_validation_set()
    # coco_filtered_remian_id = pickle.load(open("data/coco/coco_clip_filtered_ids.pickle", "rb"))
    #
    # coco_filtered_subset100 = random.sample(coco_filtered_remian_id, 100)
    # save_path = "data/coco/coco_clip_filtered_subset100.pickle"
    # with open(save_path, "wb") as f:
    #     pickle.dump(coco_filtered_subset100, f)

    # dataset = CocoCaptions_clip_filtered(root=MyPath.db_root_dir("coco_train"), annFile=MyPath.db_root_dir("coco_caption_train"),
    #                                 regenerate=False)
    dataset = CustomCocoCaptions(root=MyPath.db_root_dir("coco_val"), annFile=MyPath.db_root_dir("coco_caption_val"),
                                 custom_file="/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/jomat-code/filtering/ms_coco_captions_testset100.txt")
    dataset[0]
