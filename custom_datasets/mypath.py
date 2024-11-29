import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        coco_root = "/data/vision/torralba/datasets/coco_2017"
        sam_caption_root = "/vision-nfs/torralba/datasets/vision/sam/captions"

        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map={
            "coco_train": f"{coco_root}/train2017/",
            "coco_caption_train": f"{coco_root}/annotations/captions_train2017.json",
            "coco_val": f"{coco_root}/val2017/",
            "coco_caption_val": f"{coco_root}/annotations/captions_val2017.json",
            "sam_images": "/vision-nfs/torralba/datasets/vision/sam/images",
            "sam_captions": sam_caption_root,
            "sam_whole_filtered_ids_train": "data/filtered_sam/all_remain_ids_train.pickle",
            "sam_whole_filtered_ids_val": "data/filtered_sam/all_remain_ids_val.pickle",
            "sam_id_dict": "data/filtered_sam/all_id_dict.pickle",

            "lhq_ids_sub500": "data/LHQ500_caption/idx/subsample_500.pickle",
            "lhq_images": "data/LHQ500_caption/subsample_500",
            "lhq_captions": "data/LHQ500_caption/captions",
        }
        ret = map.get(database, None)
        if ret is None:
            raise NotImplementedError
        return ret