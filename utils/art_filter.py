# Authors: Hui Ren (rhfeiyang.github.io)

from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm

class Caption_filter:
    def __init__(self, filter_prompts=["painting", "paintings", "art", "artwork", "drawings", "sketch", "sketches", "illustration", "illustrations",
                                       "sculpture","sculptures", "installation", "printmaking", "digital art", "conceptual art", "mosaic", "tapestry",
                                       "abstract", "realism", "surrealism", "impressionism", "expressionism", "cubism", "minimalism", "baroque", "rococo",
                                       "pop art", "art nouveau", "art deco", "futurism", "dadaism",
                                        "stamp", "stamps", "advertisement", "advertisements","logo", "logos"
                                       ],):
        self.filter_prompts = filter_prompts
        self.total_count=0
        self.filter_count=[0]*len(filter_prompts)

    def reset(self):
        self.total_count=0
        self.filter_count=[0]*len(self.filter_prompts)
    def filter(self, captions):
        filter_result = []
        for caption in captions:
            words = caption[0]
            if words == None:
                filter_result.append((True, "None"))
                continue
            words = words.lower()
            words = words.split()
            filt = False
            reason=None
            for i, filter_keyword in enumerate(self.filter_prompts):
                key_len = len(filter_keyword.split())
                for j in range(len(words)-key_len+1):
                    if " ".join(words[j:j+key_len]) == filter_keyword:
                        self.filter_count[i] += 1
                        filt = True
                        reason = filter_keyword
                        break
                if filt:
                    break
            filter_result.append((filt, reason))
            self.total_count += 1
        return filter_result

class Clip_filter:
    prompt_threshold = {
        "painting": 17,
        "art": 17.5,
        "artwork": 19,
        "drawing": 15.8,
        "sketch": 17,
        "illustration": 15,
        "sculpture": 19.2,
        "installation art": 20,
        "printmaking art": 16.3,
        "digital art": 15,
        "conceptual art": 18,
        "mosaic art": 19,
        "tapestry": 16,
        "abstract art":16.5,
        "realism art": 16,
        "surrealism art": 15,
        "impressionism art": 17,
        "expressionism art": 17,
        "cubism art": 15,
        "minimalism art": 16,
        "baroque art": 17.5,
        "rococo art": 17,
        "pop art": 16,
        "art nouveau": 19,
        "art deco": 19,
        "futurism art": 16.5,
        "dadaism art": 16.5,
        "stamp": 18,
        "advertisement": 16.5,
        "logo": 15.5,
    }
    @torch.no_grad()
    def __init__(self, positive_prompt=["painting", "art", "artwork", "drawing", "sketch", "illustration",
                                        "sculpture", "installation art", "printmaking art", "digital art", "conceptual art", "mosaic art", "tapestry",
                                        "abstract art", "realism art", "surrealism art", "impressionism art", "expressionism art", "cubism art",
                                        "minimalism art", "baroque art", "rococo art",
                                        "pop art", "art nouveau", "art deco", "futurism art", "dadaism art",
                                        "stamp", "advertisement",
                                        "logo"
                                        ],
                  device="cuda"):
        self.device = device
        self.model = (CLIPModel.from_pretrained("openai/clip-vit-large-patch14")).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.positive_prompt = positive_prompt
        self.text = self.positive_prompt
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.text_encoding = self.tokenizer(self.text, return_tensors="pt", padding=True).to(device)
        self.text_features = self.model.get_text_features(**self.text_encoding)
        self.text_features = self.text_features / self.text_features.norm(p=2, dim=-1, keepdim=True)
    @torch.no_grad()
    def similarity(self, image):
        # inputs = self.processor(text=self.text, images=image, return_tensors="pt", padding=True)
        image_processed = self.image_processor(image, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
        inputs = {**self.text_encoding, **image_processed}
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image

    def get_logits(self, image):
        logits_per_image = self.similarity(image)
        return logits_per_image.cpu()

    def get_image_features(self, image):
        image_processed = self.image_processor(image, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
        image_features = self.model.get_image_features(**image_processed)
        return image_features


class Art_filter:
    def __init__(self):
        self.caption_filter = Caption_filter()
        self.clip_filter = Clip_filter()
    def caption_filt(self, dataloader):
        self.caption_filter.reset()
        dataloader.dataset.get_img = False
        dataloader.dataset.get_cap = True
        remain_ids = []
        filtered_ids = []
        for i, batch in tqdm(enumerate(dataloader)):
            captions = batch["text"]
            filter_result = self.caption_filter.filter(captions)
            for j, (filt, reason) in enumerate(filter_result):
                if filt:
                    filtered_ids.append((batch["ids"][j], reason))
                    if i%10==0:
                        print(f"Filtered caption: {captions[j]}, reason: {reason}")
                else:
                    remain_ids.append(batch["ids"][j])
        return {"remain_ids":remain_ids, "filtered_ids":filtered_ids, "total_count":self.caption_filter.total_count, "filter_count":self.caption_filter.filter_count, "filter_prompts":self.caption_filter.filter_prompts}

    def clip_filt(self, clip_logits_ckpt:dict):
        logits = clip_logits_ckpt["clip_logits"]
        ids = clip_logits_ckpt["ids"]
        text = clip_logits_ckpt["text"]
        filt_mask = torch.zeros(logits.shape[0], dtype=torch.bool)
        for i, prompt in enumerate(text):
            threshold = Clip_filter.prompt_threshold[prompt]
            filt_mask = filt_mask | (logits[:,i] >= threshold)
        filt_ids = []
        remain_ids = []
        for i, id in enumerate(ids):
            if filt_mask[i]:
                filt_ids.append(id)
            else:
                remain_ids.append(id)
        return {"remain_ids":remain_ids, "filtered_ids":filt_ids}

    def clip_feature(self, dataloader):
        dataloader.dataset.get_img = True
        dataloader.dataset.get_cap = False
        clip_features = []
        ids = []
        for i, batch in enumerate(dataloader):
            images = batch["images"]
            features = self.clip_filter.get_image_features(images).cpu()
            clip_features.append(features)
            ids.extend(batch["ids"])
        clip_features = torch.cat(clip_features)
        return {"clip_features":clip_features, "ids":ids}


    def clip_logit(self, dataloader):
        dataloader.dataset.get_img = True
        dataloader.dataset.get_cap = False
        clip_features = []
        clip_logits = []
        ids = []
        for i, batch in enumerate(dataloader):
            images = batch["images"]
            # logits = self.clip_filter.get_logits(images)
            feature = self.clip_filter.get_image_features(images)
            logits = self.clip_logit_by_feat(feature)["clip_logits"]

            clip_features.append(feature)
            clip_logits.append(logits)
            ids.extend(batch["ids"])

        clip_features = torch.cat(clip_features)
        clip_logits = torch.cat(clip_logits)
        return {"clip_features":clip_features, "clip_logits":clip_logits, "ids":ids, "text": self.clip_filter.text}

    def clip_logit_by_feat(self, feature):
        feature = feature.clone().to(self.clip_filter.device)
        feature = feature / feature.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.clip_filter.model.logit_scale.exp()
        logits = ((feature @ self.clip_filter.text_features.T) * logit_scale).cpu()
        return {"clip_logits":logits, "text": self.clip_filter.text}



if __name__ == "__main__":
    import pickle
    with open("/vision-nfs/torralba/scratch/jomat/sam_dataset/filt_result/sa_000000/clip_logits_result.pickle","rb") as f:
        result=pickle.load(f)
    feat = result['clip_features']
    logits =Art_filter().clip_logit_by_feat(feat)
    print(logits)

