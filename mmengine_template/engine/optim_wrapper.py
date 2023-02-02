from mmengine.optim import OptimWrapper

from mmengine_template.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class CustomOptimWrapper(OptimWrapper):
    ...
