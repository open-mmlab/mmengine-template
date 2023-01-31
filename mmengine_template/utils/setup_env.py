import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in `mmengine_template` into the registries.

    Args:
        init_default_scope (bool): Whether initialize the `mmengine_template`
            default scope. When `init_default_scope=True`, the global default
            scope will be set to ``mmengine_template``, and all registries will
            build modules from `mmengine_template`'s registry node. To
            understand more about the registry, please refer to `docs <https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md>`_
            Defaults to True.
    """  # noqa
    # Register transforms and cnn blocks defined in MMCV.
    import mmcv.cnn  # noqa: F401,F403
    import mmcv.transforms  # noqa: F401,F403

    import mmengine_template.datasets  # noqa: F401,F403
    import mmengine_template.engine  # noqa: F401,F403
    import mmengine_template.evaluation  # noqa: F401,F403
    import mmengine_template.models  # noqa: F401,F403
    import mmengine_template.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = (
            DefaultScope.get_current_instance() is None
            or not DefaultScope.check_instance_created('mmengine_template'))
        if never_created:
            DefaultScope.get_instance(
                'mmengine_template', scope_name='mmengine_template')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmengine_template':
            warnings.warn(
                'The current default scope '
                f'"{current_scope.scope_name}" is not "mmengine_template", '
                '`register_all_modules` will force the current'
                'default scope to be "mmengine_template". If this is not '
                'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmengine_template-{datetime.datetime.now()}'
            DefaultScope.get_instance(
                new_instance_name, scope_name='mmengine_template')
