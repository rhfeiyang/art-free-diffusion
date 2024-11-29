# Authors: Hui Ren (rhfeiyang.github.io)
import os
import sys
import numpy as np
from PIL import Image
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from custom_datasets import get_dataset
from utils.art_filter import Art_filter
import torch
from matplotlib import pyplot as plt
import math
import argparse
import socket
import time
from tqdm import tqdm
import torch
def parse_args():
    parser = argparse.ArgumentParser(description="Filter the coco dataset")
    parser.add_argument("--check", action="store_true", help="Check the complete")
    parser.add_argument("--mode", default="clip_logit", help="Filter mode: clip_logit, clip_filt, caption_filt")
    parser.add_argument("--split" , default="val", help="Dataset split, val/train")
    # parser.add_argument("--start_idx", default=0, type=int, help="Start index")
    args = parser.parse_args()
    return args

def get_feat(save_path, dataloader, filter):
    clip_feat_file = save_path
    # compute_new = False
    clip_feat={}
    if os.path.exists(clip_feat_file):
        with open(clip_feat_file, 'rb') as f:
            clip_feat = pickle.load(f)
    else:
        print(f"computing clip feat",flush=True)
        clip_feature_ret = filter.clip_feature(dataloader)
        clip_feat["image_features"] = clip_feature_ret["clip_features"]
        clip_feat["ids"] = clip_feature_ret["ids"]

        with open(clip_feat_file, 'wb') as f:
            pickle.dump(clip_feat, f)
        print(f"clip_feat_result saved to {clip_feat_file}",flush=True)
    return clip_feat

def get_clip_logit(save_root, dataloader, filter):
    feat_path = os.path.join(save_root, "clip_feat.pickle")
    clip_feat = get_feat(feat_path, dataloader, filter)
    clip_logits_file = os.path.join(save_root, "clip_logits.pickle")
    # if clip_logit:
    if os.path.exists(clip_logits_file):
        with open(clip_logits_file, 'rb') as f:
            clip_logits = pickle.load(f)
    else:
        clip_logits = filter.clip_logit_by_feat(clip_feat["image_features"])
        clip_logits["ids"] = clip_feat["ids"]
        with open(clip_logits_file, 'wb') as f:
            pickle.dump(clip_logits, f)
        print(f"clip_logits_result saved to {clip_logits_file}",flush=True)
    return clip_logits

def clip_filt(save_root, dataloader, filter):
    clip_filt_file = os.path.join(save_root, "clip_filt_result.pickle")
    if os.path.exists(clip_filt_file):
        with open(clip_filt_file, 'rb') as f:
            clip_filt_result = pickle.load(f)
    else:
        clip_logits = get_clip_logit(save_root, dataloader, filter)
        clip_filt_result = filter.clip_filt(clip_logits)
        with open(clip_filt_file, 'wb') as f:
            pickle.dump(clip_filt_result, f)
        print(f"clip_filt_result saved to {clip_filt_file}",flush=True)
    return clip_filt_result

def caption_filt(save_root, dataloader, filter):
    caption_filt_file = os.path.join(save_root, "caption_filt_result.pickle")
    if os.path.exists(caption_filt_file):
        with open(caption_filt_file, 'rb') as f:
            caption_filt_result = pickle.load(f)
    else:
        caption_filt_result = filter.caption_filt(dataloader)
        with open(caption_filt_file, 'wb') as f:
            pickle.dump(caption_filt_result, f)
        print(f"caption_filt_result saved to {caption_filt_file}",flush=True)
    return caption_filt_result

def gather_result(save_dir, dataloader, filter):
    all_remain_ids=[]
    all_remain_ids_train=[]
    all_remain_ids_val=[]
    all_filtered_id_num = 0

    clip_filt_result = clip_filt(save_dir, dataloader, filter)
    caption_filt_result = caption_filt(save_dir, dataloader, filter)

    caption_filtered_ids = [i[0] for i in caption_filt_result["filtered_ids"]]
    all_filtered_id_num += len(set(clip_filt_result["filtered_ids"]) | set(caption_filtered_ids) )
    remain_ids = set(clip_filt_result["remain_ids"]) & set(caption_filt_result["remain_ids"])
    remain_ids = list(remain_ids)
    remain_ids.sort()
    with open(os.path.join(save_dir, "remain_ids.pickle"), 'wb') as f:
        pickle.dump(remain_ids, f)
    print(f"remain_ids saved to {save_dir}/remain_ids.pickle",flush=True)
    return remain_ids

@torch.no_grad()
def main(args):
    filter = Art_filter()
    if args.mode == "caption_filt" or args.mode == "gather_result":
        filter.clip_filter = None
        torch.cuda.empty_cache()

    # caption_folder_path = "/vision-nfs/torralba/scratch/jomat/sam_dataset/PixArt-alpha/captions"
    # image_folder_path = "/vision-nfs/torralba/scratch/jomat/sam_dataset/images"
    # id_dict_dir = "/vision-nfs/torralba/scratch/jomat/sam_dataset/images/id_dict"
    # filt_dir = "/vision-nfs/torralba/scratch/jomat/sam_dataset/filt_result"

    def collate_fn(examples):
        # {"image": image, "id":id}
        ret = {}
        if "image" in examples[0]:
            pixel_values = [example["image"] for example in examples]
            ret["images"] = pixel_values
        if "caption" in examples[0]:
            # prompts = [example["caption"] for example in examples]
            prompts = []
            for example in examples:
                if isinstance(example["caption"][0], list):
                    prompts.append([" ".join(example["caption"][0])])
                else:
                    prompts.append(example["caption"])
            ret["text"] = prompts
        id = [example["id"] for example in examples]
        ret["ids"] = id
        return ret
    if args.split == "val":
        dataset = get_dataset("coco_val")["val"]
    elif args.split == "train":
        dataset = get_dataset("coco_train", get_val=False)["train"]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)

    error_files=[]



    save_root = f"/vision-nfs/torralba/scratch/jomat/sam_dataset/coco/filt/{args.split}"
    os.makedirs(save_root, exist_ok=True)

    if args.mode == "clip_feat":
        feat_path = os.path.join(save_root, "clip_feat.pickle")
        clip_feat = get_feat(feat_path, dataloader, filter)

    if args.mode == "clip_logit":
        clip_logit = get_clip_logit(save_root, dataloader, filter)

    if args.mode == "clip_filt":
        # if os.path.exists(clip_filt_file):
        #     with open(clip_filt_file, 'rb') as f:
        #         ret = pickle.load(f)
        # else:
        clip_filt_result = clip_filt(save_root, dataloader, filter)

    if args.mode == "caption_filt":
        caption_filt_result = caption_filt(save_root, dataloader, filter)

    if args.mode == "gather_result":
        filtered_result = gather_result(save_root, dataloader, filter)

    print("finished",flush=True)
    for file in error_files:
        # os.remove(file)
        print(file,flush=True)

if __name__ == "__main__":
    args = parse_args()

    log_file = "sam_filt"
    idx=0
    hostname = socket.gethostname()
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    while os.path.exists(f"{log_file}_{hostname}_check{args.check}_{now_time}_{idx}.log"):
        idx+=1

    main(args)
    # clip_logits_analysis()


