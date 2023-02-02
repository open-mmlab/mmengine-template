from mmengine.hooks import Hook

from mmengine_template.registry import HOOKS


@HOOKS.register_module()
class CustomHook(Hook):
    priority = 'NORMAL'
    ...
