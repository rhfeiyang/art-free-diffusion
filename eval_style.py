# Authors: Hui Ren (rhfeiyang.github.io)
import torch
import argparse
from inference import infer_metric
import os
import pickle
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate style metrics")
    parser.add_argument("--gen_path", type=str, required=True, help="Path to the folder contains infer_imgs.pickle and infer_prompts.txt")
    parser.add_argument("--ref_path", type=str, required=True, help="Path to the ref painting image folder")
    parser.add_argument("--save_dir", type=str, default="./", help="Path to save the results")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args.gen_path)
    print(args.ref_path)
    pred_imgs_path = os.path.join(args.gen_path, "infer_imgs.pickle")
    infer_prompts_path = os.path.join(args.gen_path, "infer_prompts.txt")
    with open(pred_imgs_path, "rb") as f:
        pred_imgs = pickle.load(f)
    with open(infer_prompts_path, "r") as f:
        infer_prompts = f.readlines()
    print("Start evaluating style metrics")
    scores = infer_metric(args.ref_path ,pred_imgs, infer_prompts, args.save_dir)
    print(scores)