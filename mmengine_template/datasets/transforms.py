"""This module is used to implement and register the custom data
transformation.

MMCV and OpenMMLab series repositories have provided rich transformation. You
can customize the transformation based on the existed ones.

The default implementation only does the register process. Users need to rename
the ``CustomTransform`` to the real name of the transformation and then
implement it.
"""

from mmcv.transforms import BaseTransform

from mmengine_template.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CustomTransform(BaseTransform):
    ...

    def transform(self, results):
        return super().transform(results)
