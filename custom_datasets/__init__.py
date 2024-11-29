from .mypath import MyPath
from copy import deepcopy
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np

def get_dataset(dataset_name, transformation=None , train_subsample:int =None, val_subsample:int = 10000, get_val=True):
    if train_subsample is not None and train_subsample<val_subsample and train_subsample!=-1:
        print(f"Warning: train_subsample is smaller than val_subsample. val_subsample will be set to train_subsample: {train_subsample}")
        val_subsample = train_subsample

    if dataset_name == "imagenet":
        from .imagenet import Imagenet1k
        train_set = Imagenet1k(data_dir = MyPath.db_root_dir(dataset_name), transform = transformation, split="train", prompt_transform=Label_prompt_transform(real=True))
    elif dataset_name == "coco_train":
        # raise NotImplementedError("Use coco_filtered instead")
        from .coco import CocoCaptions
        train_set = CocoCaptions(root=MyPath.db_root_dir("coco_train"), annFile=MyPath.db_root_dir("coco_caption_train"))
    elif dataset_name == "coco_val":
        from .coco import CocoCaptions
        train_set = CocoCaptions(root=MyPath.db_root_dir("coco_val"), annFile=MyPath.db_root_dir("coco_caption_val"))
        return {"val": train_set}

    elif dataset_name == "coco_clip_filtered":
        from .coco import CocoCaptions_clip_filtered
        train_set = CocoCaptions_clip_filtered(root=MyPath.db_root_dir("coco_train"), annFile=MyPath.db_root_dir("coco_caption_train"))
    elif dataset_name == "coco_filtered_sub100":
        from .coco import CocoCaptions_clip_filtered
        train_set = CocoCaptions_clip_filtered(root=MyPath.db_root_dir("coco_train"), annFile=MyPath.db_root_dir("coco_caption_train"), id_file=MyPath.db_root_dir("coco_clip_filtered_ids_sub100"),)
    elif dataset_name == "cifar10":
        from .cifar import CIFAR10
        train_set = CIFAR10(root=MyPath.db_root_dir("cifar10"), train=True, transform=transformation, prompt_transform=Label_prompt_transform(real=True))
    elif dataset_name == "cifar100":
        from .cifar import CIFAR100
        train_set = CIFAR100(root=MyPath.db_root_dir("cifar100"), train=True, transform=transformation, prompt_transform=Label_prompt_transform(real=True))
    elif "wikiart" in dataset_name and "/" not in dataset_name:
        from .wikiart.wikiart import Wikiart_caption
        dataset = Wikiart_caption(data_path=MyPath.db_root_dir(dataset_name))
        return {"train": dataset.subsample(train_subsample).get_dataset(), "val": deepcopy(dataset).subsample(val_subsample).get_dataset() if get_val else None}
    elif "imagepair" in dataset_name:
        from .imagepair import ImagePair
        train_set = ImagePair(folder1=MyPath.db_root_dir(dataset_name)[0], folder2=MyPath.db_root_dir(dataset_name)[1], transform=transformation).subsample(train_subsample)
    # elif dataset_name == "sam_clip_filtered":
    #     from .sam import SamDataset
    #     train_set = SamDataset(image_folder_path=MyPath.db_root_dir("sam_images"), caption_folder_path=MyPath.db_root_dir("sam_captions"), id_file=MyPath.db_root_dir("sam_ids"), transforms=transformation).subsample(train_subsample)
    elif dataset_name == "sam_whole_filtered":
        from .sam import SamDataset
        train_set = SamDataset(image_folder_path=MyPath.db_root_dir("sam_images"), caption_folder_path=MyPath.db_root_dir("sam_captions"), id_file=MyPath.db_root_dir("sam_whole_filtered_ids_train"), id_dict_file=MyPath.db_root_dir("sam_id_dict"), transforms=transformation).subsample(train_subsample)
    elif dataset_name == "sam_whole_filtered_val":
        from .sam import SamDataset
        train_set = SamDataset(image_folder_path=MyPath.db_root_dir("sam_images"), caption_folder_path=MyPath.db_root_dir("sam_captions"), id_file=MyPath.db_root_dir("sam_whole_filtered_ids_val"), id_dict_file=MyPath.db_root_dir("sam_id_dict"), transforms=transformation).subsample(train_subsample)
        return {"val": train_set}
    elif dataset_name == "lhq_sub100":
        from .lhq import LhqDataset
        train_set = LhqDataset(image_folder_path=MyPath.db_root_dir("lhq_images"), caption_folder_path=MyPath.db_root_dir("lhq_captions"), id_file=MyPath.db_root_dir("lhq_ids_sub100"), transforms=transformation)
    elif dataset_name == "lhq_sub500":
        from .lhq import LhqDataset
        train_set = LhqDataset(image_folder_path=MyPath.db_root_dir("lhq_images"), caption_folder_path=MyPath.db_root_dir("lhq_captions"), id_file=MyPath.db_root_dir("lhq_ids_sub500"), transforms=transformation)
    elif dataset_name == "lhq_sub9":
        from .lhq import LhqDataset
        train_set = LhqDataset(image_folder_path=MyPath.db_root_dir("lhq_images"), caption_folder_path=MyPath.db_root_dir("lhq_captions"), id_file=MyPath.db_root_dir("lhq_ids_sub9"), transforms=transformation)

    elif dataset_name == "custom_coco100":
        from .coco import CustomCocoCaptions
        train_set = CustomCocoCaptions(root=MyPath.db_root_dir("coco_val"), annFile=MyPath.db_root_dir("coco_caption_val"),
                           custom_file=MyPath.db_root_dir("custom_coco100_captions"), transforms=transformation)
    elif dataset_name == "custom_coco500":
        from .coco import CustomCocoCaptions
        train_set = CustomCocoCaptions(root=MyPath.db_root_dir("coco_val"), annFile=MyPath.db_root_dir("coco_caption_val"),
                           custom_file=MyPath.db_root_dir("custom_coco500_captions"), transforms=transformation)
    elif dataset_name == "laion_pop500":
        from .custom_caption import Laion_pop
        train_set = Laion_pop(anno_file=MyPath.db_root_dir("laion_pop500"), image_root=MyPath.db_root_dir("laion_images"), transform=transformation)

    elif dataset_name == "laion_pop500_first_sentence":
        from .custom_caption import Laion_pop
        train_set = Laion_pop(anno_file=MyPath.db_root_dir("laion_pop500_first_sentence"), image_root=MyPath.db_root_dir("laion_images"), transform=transformation)


    else:
        try:
            train_set = load_dataset('imagefolder', data_dir = dataset_name, split="train")
            val_set = deepcopy(train_set)
            if val_subsample is not None and val_subsample != -1:
                val_set = val_set.shuffle(seed=0).select(range(val_subsample))
            return {"train": train_set, "val": val_set if get_val else None}
        except:
            raise ValueError(f"dataset_name {dataset_name} not found.")
    return {"train": train_set, "val": deepcopy(train_set).subsample(val_subsample) if get_val else None}


class MergeDataset(Dataset):
    @staticmethod
    def get_merged_dataset(dataset_names:list, transformation=None, train_subsample:int =None, val_subsample:int = 10000):
        train_datasets = []
        val_datasets = []
        for dataset_name in dataset_names:
            datasets = get_dataset(dataset_name, transformation, train_subsample, val_subsample)
            train_datasets.append(datasets["train"])
            val_datasets.append(datasets["val"])
        train_datasets = MergeDataset(train_datasets).subsample(train_subsample)
        val_datasets = MergeDataset(val_datasets).subsample(val_subsample)
        return {"train": train_datasets, "val": val_datasets}

    def __init__(self, datasets:list):
        self.datasets = datasets
        self.column_names = self.datasets[0].column_names
        # self.ids = []
        # start = 0
        # for dataset in self.datasets:
        #     self.ids += [i+start for i in dataset.ids]
    def define_resolution(self, resolution: int):
        for dataset in self.datasets:
            dataset.define_resolution(resolution)

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
    def __getitem__(self, index):
        for i,dataset in enumerate(self.datasets):
            if index < len(dataset):
                ret = dataset[index]
                ret["id"] = index
                ret["dataset"] = i
                return ret
            index -= len(dataset)
        raise IndexError

    def subsample(self, num:int):
        if num is None:
            return self
        dataset_ratio = np.array([len(dataset) for dataset in self.datasets]) / len(self)
        new_datasets = []
        for i, dataset in enumerate(self.datasets):
            new_datasets.append(dataset.subsample(int(num*dataset_ratio[i])))
        return MergeDataset(new_datasets)

    def with_transform(self, transform):
        for dataset in self.datasets:
            dataset.with_transform(transform)
        return self

