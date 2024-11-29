# Authors: Hui Ren (rhfeiyang.github.io)
import argparse
from huggingface_hub import hf_hub_download
import zipfile
import os
name_map = {
    "art_styles":"Art_styles.zip",
    "art_adapters":"Art_adapters.zip",
    "filtered_sam":"filtered_sam.zip",
    "csd":"pytorch_model.bin",
    "laion_pop500": "laion_pop500",
}


def download_data(data_list, save_root="./"):
    for data in data_list:
        if "csd" in data:
            repo_id = "tomg-group-umd/CSD-ViT-L"
            hf_hub_download(repo_id=repo_id,repo_type="model", filename="pytorch_model.bin", local_dir=os.path.join(save_root,"weights"))
            os.rename(os.path.join(save_root,"weights","pytorch_model.bin"), os.path.join(save_root,"weights","CSD-checkpoint.pth"))
        elif data == "laion_pop500":
            repo_id = "rhfeiyang/art-free-diffusion_resources"
            hf_hub_download(repo_id=repo_id,repo_type="dataset", subfolder="laion_pop500", local_dir=save_root, filename="laion_pop500_images.zip")
            hf_hub_download(repo_id=repo_id,repo_type="dataset", subfolder="laion_pop500", local_dir=save_root, filename="laion_pop500.csv")
            hf_hub_download(repo_id=repo_id,repo_type="dataset", subfolder="laion_pop500", local_dir=save_root, filename="laion_pop500_first_sentence.csv")

            file_path=os.path.join(save_root,"laion_pop500", "laion_pop500_images.zip")
            print(f"Extracting {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(save_root,"laion_pop500"))
        else:
            repo_id = "rhfeiyang/art-free-diffusion_resources"
            name = name_map[data]
            hf_hub_download(repo_id=repo_id,repo_type="dataset", filename=name, local_dir=save_root)
            file_path=os.path.join(save_root,name)
            if ".zip" in name:
                print(f"Extracting {file_path}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(save_root)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", type=str, required=True, nargs='+', choices=["art_styles","art_adapters", "filtered_sam", "csd", "laion_pop500",])
    args = parser.parse_args()
    download_data(args.data)

