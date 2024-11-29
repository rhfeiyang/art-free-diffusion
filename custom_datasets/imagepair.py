# Authors: Hui Ren (rhfeiyang.github.io)
import random

import torch.utils.data as data
from PIL import Image
import os
import torch
# from tqdm import tqdm
class ImageSet(data.Dataset):
    def __init__(self, folder , transform=None, keep_in_mem=True, caption=None):
        self.path = folder
        self.transform = transform
        self.caption_path = None
        self.images = []
        self.captions = []
        self.keep_in_mem = keep_in_mem

        if not isinstance(folder, list):
            self.image_files = [file for file in os.listdir(folder) if file.endswith((".png",".jpg"))]
            self.image_files.sort()
        else:
            self.images = folder

        if not isinstance(caption, list):
            if caption not in [None, "", "None"]:
                self.caption_path = caption
                self.caption_files = [os.path.join(caption, file.replace(".png", ".txt").replace(".jpg", ".txt")) for file in self.image_files]
                self.caption_files.sort()
        else:
            self.caption_path = True
            self.captions = caption
        # get all the image files png/jpg


        if keep_in_mem:
            if len(self.images) == 0:
                for file in self.image_files:
                    img = self.load_image(os.path.join(self.path, file))
                    self.images.append(img)
            if len(self.captions) == 0:
                if self.caption_path is not None:
                    self.captions = []
                    for file in self.caption_files:
                        caption = self.load_caption(file)
                        self.captions.append(caption)
        else:
            self.images = None

    def limit_num(self, n):
        raise NotImplementedError
        assert n <= len(self), f"n should be less than the length of the dataset {len(self)}"
        self.image_files = self.image_files[:n]
        self.caption_files = self.caption_files[:n]
        if self.keep_in_mem:
            self.images = self.images[:n]
            self.captions = self.captions[:n]
        print(f"Dataset limited to {n}")

    def __len__(self):
        if len(self.images) != 0:
            return len(self.images)
        else:
            return len(self.image_files)

    def load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img

    def load_caption(self, path):
        with open(path, 'r') as f:
            caption = f.readlines()
        caption = [line.strip() for line in caption if len(line.strip()) > 0]
        return caption

    def __getitem__(self, index):
        if len(self.images) != 0:
            img = self.images[index]
        else:
            img = self.load_image(os.path.join(self.path, self.image_files[index]))

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.caption_path is not None or len(self.captions) != 0:
            if len(self.captions) != 0:
                caption = self.captions[index]
            else:
                caption = self.load_caption(self.caption_files[index])
            ret= {"image": img, "caption": caption, "id": index}
        else:
            ret= {"image": img, "id": index}
        if self.transform is not None:
            ret = self.transform(ret)
        return ret

    def subsample(self, n: int = 10):
        if n is None or n == -1:
            return self
        ori_len = len(self)
        assert n <= ori_len
        # equal interval subsample
        ids = self.image_files[::ori_len // n][:n]
        self.image_files = ids
        if self.keep_in_mem:
            self.images = self.images[::ori_len // n][:n]
        print(f"Dataset subsampled from {ori_len} to {len(self)}")
        return self

    def with_transform(self, transform):
        self.transform = transform
        return self
    @staticmethod
    def collate_fn(examples):
        images = [example["image"] for example in examples]
        ids = [example["id"] for example in examples]
        if "caption" in examples[0]:
            captions = [random.choice(example["caption"]) for example in examples]
            return {"images": images, "captions": captions, "id": ids}
        else:
            return {"images": images, "id": ids}


class ImagePair(ImageSet):
    def __init__(self, folder1, folder2, transform=None, keep_in_mem=True):
        self.path1 = folder1
        self.path2 = folder2
        self.transform = transform
        # get all the image files png/jpg
        self.image_files = [file for file in os.listdir(folder1) if file.endswith(".png") or file.endswith(".jpg")]
        self.image_files.sort()
        self.keep_in_mem = keep_in_mem
        if keep_in_mem:
            self.images = []
            for file in self.image_files:
                img1 = self.load_image(os.path.join(self.path1, file))
                img2 = self.load_image(os.path.join(self.path2, file))
                self.images.append((img1, img2))
        else:
            self.images = None

    def __getitem__(self, index):
        if self.keep_in_mem:
            img1, img2 = self.images[index]
        else:
            img1 = self.load_image(os.path.join(self.path1, self.image_files[index]))
            img2 = self.load_image(os.path.join(self.path2, self.image_files[index]))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return {"image1": img1, "image2": img2, "id": index}



    @staticmethod
    def collate_fn(examples):
        images1 = [example["image1"] for example in examples]
        images2 = [example["image2"] for example in examples]
        # images1 = torch.stack(images1)
        # images2 = torch.stack(images2)
        ids = [example["id"] for example in examples]
        return {"image1": images1, "image2": images2, "id": ids}

    def push_to_huggingface(self, hug_folder):
        from datasets import Dataset
        from datasets import Image as HugImage
        photo_path = [os.path.join(self.path1, file) for file in self.image_files]
        sketch_path = [os.path.join(self.path2, file) for file in self.image_files]
        dataset = Dataset.from_dict({"photo": photo_path, "sketch": sketch_path, "file_name": self.image_files})
        dataset = dataset.cast_column("photo", HugImage())
        dataset = dataset.cast_column("sketch", HugImage())
        dataset.push_to_hub(hug_folder, private=True)

class ImageClass(ImageSet):
    def __init__(self, folders: list, transform=None, keep_in_mem=True):
        self.paths = folders
        self.transform = transform
        # get all the image files png/jpg
        self.image_files = []
        self.keep_in_mem = keep_in_mem
        for i, folder in enumerate(folders):
            self.image_files+=[(os.path.join(folder, file), i) for file in os.listdir(folder) if file.endswith(".png") or file.endswith(".jpg")]
        if keep_in_mem:
            self.images = []
            print("Loading images to memory")
            for file in self.image_files:
                img = self.load_image(file[0])
                self.images.append((img, file[1]))
            print("Loading images to memory done")
        else:
            self.images = None

    def __getitem__(self, index):
        if self.keep_in_mem:
            img, label = self.images[index]
        else:
            img_path, label = self.image_files[index]
            img = self.load_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": label, "id": index}

    @staticmethod
    def collate_fn(examples):
        images = [example["image"] for example in examples]
        labels = [example["label"] for example in examples]
        ids = [example["id"] for example in examples]
        return {"images": images, "labels":labels, "id": ids}


if __name__ == "__main__":
    # dataset = ImagePair("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/clip_filtered_remain_50",
    #                     "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/sketch_50",keep_in_mem=False)
    # dataset.push_to_huggingface("rhfeiyang/photo-sketch-pair-50")



    dataset = ImagePair("/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/clip_filtered_remain_500",
                        "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/imgFolder/sketch_500",
                        keep_in_mem=True)
    # dataset.push_to_huggingface("rhfeiyang/photo-sketch-pair-500")
    # ret = dataset[0]
    # print(len(dataset))
    import torch
    from torchvision import transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = dataset.with_transform(train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=ImagePair.collate_fn)
    ret = dataloader.__iter__().__next__()
    pass