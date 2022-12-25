from mmcv.transforms import BaseTransform

from mmengine_template.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CustomTransform(BaseTransform):
    ...

    def transform(self, results):
        return super().transform(results)
