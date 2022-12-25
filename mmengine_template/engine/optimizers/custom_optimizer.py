from torch.optim import Optimizer

from mmengine_template.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class CustomOptimizer(Optimizer):
    ...
