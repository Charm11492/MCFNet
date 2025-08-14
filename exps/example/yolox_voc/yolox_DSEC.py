# encoding: utf-8
import os
import sys
# sys.path.append('/root/data1/code/YOLOX-main/YOLOX-main/yolox')
from yolox.data import get_yolox_datadir
from yolox.data import COCODataset
from yolox.exp import Exp as MyExp
from torchvision import transforms
from yolox.data.data_augment import ValTransform
# from yolox.data import CSVDataset_event, collater, AspectRatioBasedSampler, \
#     Augmenter, \
#     Normalizer

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 8
        # self.depth = 0.33
        # self.width = 0.50
        self.depth = 0.67
        self.width = 0.75
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 0.9
        self.mixup_prob = 0.5
        self.hsv_prob = 0
        self.flip_prob = 0
        self.data_dir='/root/data1/standby/YOLOX-mamba/datasets'

        self.root_event_dir='/root/data1/dataset/DSEC/iwe_2'
        self.root_img_dir='/root/data1/dataset/DSEC/train/images'
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        # from yolox.data import DSECrgb, TrainTransform
   
        # dataset=DSECrgb(
        #     data_dir=self.root_img_dir,
        #     # json_file='/root/data1/code/YOLOX-main/YOLOX-main/datasets/new/train_sub_2class.json',
        #     # json_file='/root/data1/code/YOLOX-main/YOLOX-main/datasets/train_8class.json',
        #     json_file='/root/data1/code/YOLOX-main/YOLOX-main/datasets/train_8classhigh.json',
        #     img_size=self.test_size,
        #     preproc=TrainTransform(
        #         max_labels=50,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob)
        # )

        # return dataset
        from yolox.data import COCODataset, TrainTransform
   
        dataset=COCODataset(
            data_dir=self.root_img_dir,
            event_dir = self.root_event_dir,
            # json_file='/root/data1/standby/YOLOX-mamba/datasets/train_2class_960*1280.json',
            # json_file='/root/data1/standby/YOLOX-mamba/datasets/new/train_8class_960*1280.json',
            json_file='/root/data1/standby/YOLOX-mamba/datasets/train_8class_960*1280.json',
      
            img_size=self.test_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob)
        )

        return dataset
        
    def get_eval_dataset(self, **kwargs):

        # from yolox.data import DSECrgb, TrainTransform
        # legacy = kwargs.get("legacy", False)
        # return DSECrgb(
        #     data_dir=self.root_img_dir,
        #     # json_file='/root/data1/code/YOLOX-main/YOLOX-main/datasets/new/test_sub_2class.json',
        #     # json_file='/root/data1/code/YOLOX-main/YOLOX-main/datasets/test_8class.json',
        #     json_file='/root/data1/code/YOLOX-main/YOLOX-main/datasets/test_8classhigh.json',
    
        #     img_size=self.test_size,
        #     preproc=ValTransform(legacy=legacy)
        # )
        from yolox.data import COCODataset, TrainTransform
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.root_img_dir,
            event_dir = self.root_event_dir,
            # json_file='/root/data1/standby/YOLOX-mamba/datasets/test_2class_960*1280.json',
            # json_file='/root/data1/standby/YOLOX-mamba/datasets/new/test_8class_960*1280.json',
            json_file='/root/data1/standby/YOLOX-mamba/datasets/test_8class_960*1280.json',
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy)
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
