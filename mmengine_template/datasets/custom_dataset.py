from mmengine.dataset import BaseDataset

from mmengine_template.registry import DATASETS


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    ...