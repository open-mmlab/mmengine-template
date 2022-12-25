""" Implement the custom visualizer in this package.

Without using BaseDataElement, the user does not need to implement
XXXLocalVisualizer themselves. They can simply use the built-in
LocalVisualizer provided by MMEngine at the location where visualization
is required.

If the user has extracted some highly useful visualization functions for their
own tasks, they can also implement a custom CustomVisualizer here.
"""

from mmengine.visualization import Visualizer

from mmengine_template.registry import VISUALIZERS


@VISUALIZERS.register_module()
class CustomVisualizer(Visualizer):
    """ Implement the custom visualizer here. """
    ...
