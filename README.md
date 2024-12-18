# Art-Free Generative Models: Art Creation Without Graphic Art Knowledge
<a href="https://joaanna.github.io/art-free-diffusion"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2412.00176"><img src="https://img.shields.io/badge/arXiv-2412.00176-b31b1b.svg"></a>
<a href="https://huggingface.co/rhfeiyang/art-free-diffusion-v1"><img src="https://img.shields.io/badge/Art_Free-Diffusion_v1-purple"></a>
<a href="https://huggingface.co/datasets/rhfeiyang/Art-Free-SAM"><img src="https://img.shields.io/badge/Art_Free-SAM-green"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>

![teaser](docs/Teaser.jpg)

<br>
<p align="center">

> <a href="https://joaanna.github.io/art-free-diffusion">**Art-Free Generative Models: Art Creation Without Graphic Art Knowledge**</a>
>
<a href="https://rhfeiyang.github.io/" target="_blank">Hui Ren*</a>,
<a href="https://joaanna.github.io/" target="_blank">Joanna Materzynska*</a>,
<a href="https://rohitgandikota.github.io/" target="_blank">Rohit Gandikota</a>,
<a href="https://baulab.info/" target="_blank">David Bau</a>,
<a href="https://groups.csail.mit.edu/vision/torralbalab/" target="_blank">Antonio Torralba</a>

(* indicates equal contribution)


> We explore the question: ``How much prior art knowledge is needed to create art?". To find out, we designed a text-to-image generation model that skips training on art-related content entirely. Then, we developed a straightforward method to create an "art adapter," which learns artistic styles using just a handful of examples. Our experiments reveal that the art generated this way is rated by users as on par with pieces from models trained on massive, art-heavy datasets. Finally, through data attribution techniques, we illustrate how examples from both artistic and non-artistic datasets contributed to the creation of new artistic styles.
</p>


# Colab Demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rhfeiyang/art-free-diffusion/blob/master/demo.ipynb)

See `demo.ipynb` for a demo of our Art-Free Diffusion model and Art Adapter.

# Setup
To set up your python environment:
``` shell
git clone https://github.com/rhfeiyang/art-free-diffusion.git
cd art-free-diffusion
conda env create -n diffusion -f environment.yml
conda activate diffusion
```

# Data preparation
## Art-free SAM
Download original SA-1B dataset from [here](https://ai.meta.com/datasets/segment-anything-downloads/) and extract by keeping the split folder structure. Download caption dataset(SAM-LLaVA-Captions10M) from [here](https://huggingface.co/datasets/PixArt-alpha/SAM-LLaVA-Captions10M/tree/main) and extract. The folder structure should be like:
```
sam_dataset
├── captions
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
├── images
│   ├── sa_000000
│     ├── 0.jpg
│     ├── 1.jpg
│     └── ...
│   ├── sa_000001
│     ├── 0.jpg
│     ├── 1.jpg
│     └── ...
│   ├── ...
│   └── sa_000999
└── 
```
Then specify the dataset roots in `custom_datasets/mypath.py`(sam_images, sam_captions).

To Download our Art-Free SAM ids:
``` shell
cd data
python download.py -d Art-Free-SAM
cd ..
```


## Artistic Style Dataset
Download our selected artistic style dataset (obtained from wikiart) by:
``` shell
cd data
python download.py -d art_styles
cd ..
```

## Laion-pop500
Download our 500 images with annotations by:
``` shell
cd data
python download.py -d laion_pop500
cd ..
```

[//]: # (# Art filtering)

[//]: # (## Caption level filtering)

[//]: # (``` shell)

[//]: # (python custom_datasets/filt/sam_filt.py --mode caption_filt)

[//]: # (```)

[//]: # (## Image level filtering)

[//]: # (``` shell)

[//]: # (python custom_datasets/filt/sam_filt.py --mode clip_logit)

[//]: # (python custom_datasets/filt/sam_filt.py --mode clip_filt)

[//]: # (```)

[//]: # ()
[//]: # (## Gather filtering results)

[//]: # (After caption level and image level filtering, we finally gather all results:)

[//]: # (``` shell)

[//]: # (python custom_datasets/filt/sam_filt.py --mode gather_result)

[//]: # (```)


# Train artistic adapter



## Purpose and Ethical Use
- Users should only provide content that they own or have rights to, including their own creative works or photographic art.
- The tool is not designed to learn from or replicate the styles of external artists, guaranteeing that any generated models are based solely on real-world data or the user’s personal input.

## Model Zoo
Download the pre-trained 17 art adapters(Derain, Corot, Matisse, Klimt, Picasso, Andy, Richter, Hokusai, Monet, Van Gogh, ...) by:

``` shell
cd data
python download.py art_adapters
cd ..
```

## Train
To train an Art Adapter, specify the style folder and running:
``` shell
python train_artistic.py --style_folder <style_folder> --save_path <save_path>
```

For example, to train an adapter on Derain's art style:
``` shell
python train_artistic.py --style_folder data/Art_styles/andre-derain/fauvism/subset1 --save_path <save_path>
```
The trained adaptor will be saved in the format like `<save_path>/adapter_alpha1.0_rank1_all_up_1000steps.pt`

Optional arguments:
- `--style_folder`: the path to the test-time artistic style dataset
- `--save_path`: the path to save the trained adaptor
- `--rank`: the rank of the adaptor
- `--iterations`: the number of iterations to train the adaptor



# Inference with an Art Adapter

## Art Generation from prompts

``` shell
python inference.py --lora_weights <lora_location> --from_scratch --start_noise -1 --infer_prompts <prompts-or-file> --save_dir <save_location>
```
For example, to generate an image from the prompt "Sunset over the ocean with waves and rocks":
``` shell
python inference.py --lora_weights <lora_location> --from_scratch --start_noise -1 --infer_prompts "Sunset over the ocean with waves and rocks" --save_dir <save_location>
```
And to generate from a `prompts.txt` containing prompts each line:
``` shell
python inference.py --lora_weights <lora_location> --from_scratch --start_noise -1 --infer_prompts prompts.txt --save_dir <save_location>
```
Optional arguments:
- `--seed`: the seed number for random generation, default is not deterministic
- `--start_noise`: the time step that adaptor starts to be incorporated into the generation, -1 for all steps
- `--no_load`: whether to force generation even if there is an existing output in the save directory
- `--infer_prompts`: it can receive a list of string input or `.txt`/`.csv` input. For csv, specify caption column by name `caption`, and `seed` column with random seeds for each caption.



## Image Stylization
For image stylization, the arg `--val_set` specifies the name of the validation set. For example,
``` shell
python inference.py --lora_weights <lora_location> --start_noise 800 --val_set laion_pop500 --save_dir <save_location>
```
Optional arguments:
- `--val_set`: the name of the validation set, supports `laion_pop500`, `laion_pop500_first_sentence`, `lhq500`
- `--seed`: the seed number for random generation, default is not deterministic
- `--start_noise`: the time step that adaptor starts to be incorporated into the generation, -1 is for all steps. For positive values (0-1000), the larger the value, the more the adaptor is incorporated into the generation.
- `--no_load`: whether to force generation even if there is an existing output in the save directory

## Metric evaluation
First download the CSD model by:
``` shell
cd data
python download.py csd
cd ..
```

Then just add `--ref_image_folder <path_to_style_set>` to the inference command. For example,
``` shell
python inference.py --lora_weights <lora_location> --start_noise 800 --val_set laion_pop500 --save_dir <save_location> --ref_image_folder data/Art_styles/andre-derain/fauvism/subset1
```


## Citation
If you find this useful for your research, please cite the following:
```bibtex
@misc{ren2024art-free,
    title={Art-Free Generative Models: Art Creation Without Graphic Art Knowledge},
    author={Hui Ren and Joanna Materzynska and Rohit Gandikota and David Bau and Antonio Torralba},
    year={2024},
    eprint={2412.00176},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.00176},
}
```