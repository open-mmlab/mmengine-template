from mmeval import BaseMetric

from mmengine_template.registry import METRICS


@METRICS.register_module()
class CustomMetric(BaseMetric):
    def process(self, *args, **kwargs):
        # Need to call self.add
        ...
    
    def add(self, predictions, labels):
        ...
