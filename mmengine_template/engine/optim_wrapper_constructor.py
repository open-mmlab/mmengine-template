from mmengine.optim import DefaultOptimWrapperConstructor

from mmengine_template.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CustomOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    ...
