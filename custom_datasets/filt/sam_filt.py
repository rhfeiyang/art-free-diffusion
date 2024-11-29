# Authors: Hui Ren (rhfeiyang.github.io)
import os
import sys
import numpy as np
from PIL import Image
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from custom_datasets.sam import SamDataset
from utils.art_filter import Art_filter
import torch
from matplotlib import pyplot as plt
import math
import argparse
import socket
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Filter the sam dataset")
    parser.add_argument("--check", action="store_true", help="Check the complete")
    parser.add_argument("--mode", default="clip_logit",  choices=["clip_logit_update","clip_logit", "clip_filt", "caption_filt", "gather_result","caption_flit_append"])
    parser.add_argument("--start_idx", default=0, type=int, help="Start index")
    parser.add_argument("--end_idx", default=9e10, type=int, help="Start index")
    args = parser.parse_args()
    return args
@torch.no_grad()
def main(args):
    filter = Art_filter()
    if args.mode == "caption_filt" or args.mode == "gather_result":
        filter.clip_filter = None
        torch.cuda.empty_cache()

    caption_folder_path = "/afs/csail.mit.edu/u/h/huiren/code/diffusion/stable_diffusion/clip_dissection/SAM/subset/captions"
    image_folder_path = "/vision-nfs/torralba/scratch/jomat/sam_dataset/nfs-data/sam/images"
    id_dict_dir = "/vision-nfs/torralba/scratch/jomat/sam_dataset/sam_ids/8.16/id_dict"
    filt_dir = "/vision-nfs/torralba/scratch/jomat/sam_dataset/filt_result"
    def collate_fn(examples):
        # {"image": image, "id":id}
        ret = {}
        if "image" in examples[0]:
            pixel_values = [example["image"] for example in examples]
            ret["images"] = pixel_values
        if "text" in examples[0]:
            prompts = [example["text"] for example in examples]
            ret["text"] = prompts
        id = [example["id"] for example in examples]
        ret["ids"] = id
        return ret
    error_files=[]
    val_set = ["sa_000000"]
    result_check_set = ["sa_000020"]
    all_remain_ids=[]
    all_remain_ids_train=[]
    all_remain_ids_val=[]
    all_filtered_id_num = 0
    remain_feat_num = 0
    remain_caption_num = 0
    filter_feat_num = 0
    filter_caption_num = 0
    for idx,file in tqdm(enumerate(sorted(os.listdir(id_dict_dir)))):
        if idx < args.start_idx or idx >= args.end_idx:
            continue
        if file.endswith(".pickle") and not file.startswith("all"):
            print("=====================================")
            print(file,flush=True)
            save_dir = os.path.join(filt_dir, file.replace("_id_dict.pickle", ""))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            id_dict_file = os.path.join(id_dict_dir, file)
            with open(id_dict_file, 'rb') as f:
                id_dict = pickle.load(f)
            ids = list(id_dict.keys())
            dataset = SamDataset(image_folder_path, caption_folder_path, id_file=ids, id_dict_file=id_dict_file)
            # dataset = SamDataset(image_folder_path, caption_folder_path, id_file=[10061410, 10076945, 10310013,1042012, 4487809, 4541052], id_dict_file="/vision-nfs/torralba/scratch/jomat/sam_dataset/images/id_dict/all_id_dict.pickle")
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)
            clip_logits = None
            clip_logits_file = os.path.join(save_dir, "clip_logits_result.pickle")
            clip_filt_file = os.path.join(save_dir, "clip_filt_result.pickle")
            caption_filt_file = os.path.join(save_dir, "caption_filt_result.pickle")

            if args.mode == "clip_feat":
                compute_new = False
                clip_logits = {}
                if os.path.exists(clip_logits_file):
                    with open(clip_logits_file, 'rb') as f:
                        clip_logits = pickle.load(f)
                    if "image_features" not in clip_logits:
                        compute_new = True
                else:
                    compute_new=True
                if compute_new:
                    if clip_logits == '':
                        clip_logits = {}
                    print(f"compute clip_feat {file}",flush=True)
                    clip_feature_ret = filter.clip_feature(dataloader)
                    clip_logits["image_features"] = clip_feature_ret["clip_features"]
                    if "ids" in clip_logits:
                        assert clip_feature_ret["ids"] == clip_logits["ids"]
                    else:
                        clip_logits["ids"] = clip_feature_ret["ids"]

                    with open(clip_logits_file, 'wb') as f:
                        pickle.dump(clip_logits, f)
                    print(f"clip_feat_result saved to {clip_logits_file}",flush=True)
                else:
                    print(f"skip {clip_logits_file}",flush=True)

            if args.mode == "clip_logit":
            # if clip_logit:
                if os.path.exists(clip_logits_file):
                    try:
                        with open(clip_logits_file, 'rb') as f:
                            clip_logits = pickle.load(f)
                    except:
                        continue
                    skip = True
                    if args.check and clip_logits=="":
                        skip = False

                else:
                    skip = False
                # skip = False
                if not skip:
                    # os.makedirs(os.path.join(save_dir, "tmp"), exist_ok=True)
                    with open(clip_logits_file, 'wb') as f:
                        pickle.dump("", f)
                    try:
                        clip_logits = filter.clip_logit(dataloader)
                    except:
                        print(f"Error in clip_logit {file}",flush=True)
                        continue
                    with open(clip_logits_file, 'wb') as f:
                        pickle.dump(clip_logits, f)
                    print(f"clip_logits_result saved to {clip_logits_file}",flush=True)
                else:
                    print(f"skip {clip_logits_file}",flush=True)

            if args.mode == "clip_logit_update":
                if os.path.exists(clip_logits_file):
                    with open(clip_logits_file, 'rb') as f:
                        clip_logits = pickle.load(f)
                else:
                    print(f"{clip_logits_file} not exist",flush=True)
                    continue
                if clip_logits == "":
                    print(f"skip {clip_logits_file}",flush=True)
                    continue
                ret = filter.clip_logit_by_feat(clip_logits["clip_features"])
                # assert (clip_logits["clip_logits"] - ret["clip_logits"]).abs().max() < 0.01
                clip_logits["clip_logits"] = ret["clip_logits"]
                clip_logits["text"] = ret["text"]
                with open(clip_logits_file, 'wb') as f:
                    pickle.dump(clip_logits, f)


            if args.mode == "clip_filt":
                # if os.path.exists(clip_filt_file):
                #     with open(clip_filt_file, 'rb') as f:
                #         ret = pickle.load(f)
                # else:

                if clip_logits is None:
                    try:
                        with open(clip_logits_file, 'rb') as f:
                            clip_logits = pickle.load(f)
                    except:
                        print(f"Error in loading {clip_logits_file}",flush=True)
                        error_files.append(clip_logits_file)
                        continue
                    if clip_logits == "":
                        print(f"skip {clip_logits_file}",flush=True)
                        error_files.append(clip_logits_file)
                        continue
                clip_filt_result = filter.clip_filt(clip_logits)
                with open(clip_filt_file, 'wb') as f:
                    pickle.dump(clip_filt_result, f)
                print(f"clip_filt_result saved to {clip_filt_file}",flush=True)

            if args.mode == "caption_filt":
                if os.path.exists(caption_filt_file):
                    try:
                        with open(caption_filt_file, 'rb') as f:
                            ret = pickle.load(f)
                    except:
                        continue
                    skip = True
                    if args.check and ret=="":
                        skip = False
                        # os.remove(caption_filt_file)
                        print(f"empty {caption_filt_file}",flush=True)
                        # skip = True
                else:
                    skip = False
                if not skip:
                    with open(caption_filt_file, 'wb') as f:
                        pickle.dump("", f)
                    # try:
                    ret = filter.caption_filt(dataloader)
                    # except:
                    #     print(f"Error in filtering {file}",flush=True)
                    #     continue
                    with open(caption_filt_file, 'wb') as f:
                        pickle.dump(ret, f)
                    print(f"caption_filt_result saved to {caption_filt_file}",flush=True)
                else:
                    print(f"skip {caption_filt_file}",flush=True)

            if args.mode == "caption_flit_append":
                if not os.path.exists(caption_filt_file):
                    print(f"{caption_filt_file} not exist",flush=True)
                    continue
                with open(caption_filt_file, 'rb') as f:
                    old_caption_filt_result = pickle.load(f)
                skip = True
                for i in filter.caption_filter.filter_prompts:
                    if i not in old_caption_filt_result["filter_prompts"]:
                        skip = False
                        break
                if skip:
                    print(f"skip {caption_filt_file}",flush=True)
                    continue
                old_remain_ids = old_caption_filt_result["remain_ids"]
                new_dataset = SamDataset(image_folder_path, caption_folder_path, id_file=old_remain_ids, id_dict_file=id_dict_file)
                new_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=collate_fn)
                ret = filter.caption_filt(new_dataloader)
                old_caption_filt_result["remain_ids"] = ret["remain_ids"]
                old_caption_filt_result["filtered_ids"].extend(ret["filtered_ids"])
                new_filter_count = ret["filter_count"].copy()
                for i in range(len(old_caption_filt_result["filter_count"])):
                    new_filter_count[i] += old_caption_filt_result["filter_count"][i]

                old_caption_filt_result["filter_count"] = new_filter_count
                old_caption_filt_result["filter_prompts"] = ret["filter_prompts"]
                with open(caption_filt_file, 'wb') as f:
                    pickle.dump(old_caption_filt_result, f)



            if args.mode == "gather_result":
                with open(clip_filt_file, 'rb') as f:
                    clip_filt_result = pickle.load(f)
                with open(caption_filt_file, 'rb') as f:
                    caption_filt_result = pickle.load(f)
                caption_filtered_ids = [i[0] for i in caption_filt_result["filtered_ids"]]
                all_filtered_id_num += len(set(clip_filt_result["filtered_ids"]) | set(caption_filtered_ids) )

                remain_feat_num += len(clip_filt_result["remain_ids"])
                remain_caption_num += len(caption_filt_result["remain_ids"])
                filter_feat_num += len(clip_filt_result["filtered_ids"])
                filter_caption_num += len(caption_filtered_ids)

                remain_ids = set(clip_filt_result["remain_ids"]) & set(caption_filt_result["remain_ids"])
                remain_ids = list(remain_ids)
                remain_ids.sort()
                # with open(os.path.join(save_dir, "remain_ids.pickle"), 'wb') as f:
                #     pickle.dump(remain_ids, f)
                # print(f"remain_ids saved to {save_dir}/remain_ids.pickle",flush=True)
                all_remain_ids.extend(remain_ids)
                if file.replace("_id_dict.pickle","") in val_set:
                    all_remain_ids_val.extend(remain_ids)
                else:
                    all_remain_ids_train.extend(remain_ids)
    if args.mode == "gather_result":
        print(f"filtered ids: {all_filtered_id_num}",flush=True)
        print(f"remain feat num: {remain_feat_num}",flush=True)
        print(f"remain caption num: {remain_caption_num}",flush=True)
        print(f"filter feat num: {filter_feat_num}",flush=True)
        print(f"filter caption num: {filter_caption_num}",flush=True)
        all_remain_ids.sort()
        with open(os.path.join(filt_dir, "all_remain_ids.pickle"), 'wb') as f:
            pickle.dump(all_remain_ids, f)
        with open(os.path.join(filt_dir, "all_remain_ids_train.pickle"), 'wb') as f:
            pickle.dump(all_remain_ids_train, f)
        with open(os.path.join(filt_dir, "all_remain_ids_val.pickle"), 'wb') as f:
            pickle.dump(all_remain_ids_val, f)

        print(f"all_remain_ids saved to {filt_dir}/all_remain_ids.pickle",flush=True)
        print(f"all_remain_ids_train saved to {filt_dir}/all_remain_ids_train.pickle",flush=True)
        print(f"all_remain_ids_val saved to {filt_dir}/all_remain_ids_val.pickle",flush=True)

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


