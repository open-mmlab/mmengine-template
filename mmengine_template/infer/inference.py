# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import torch
from mmcv import imread, imwrite
from mmengine.device import get_device
from mmengine.infer import BaseInferencer
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer


class RetinaFaceInferencer(BaseInferencer):
    visualize_kwargs = {'vis_thresh'}

    def __init__(self, *args, save_path=None, **kwargs):
        if save_path is not None:
            mkdir_or_exist(save_path)
        self.save_path = save_path
        super().__init__(*args, **kwargs)

    def _init_visualizer(self, cfg):
        return Visualizer('retinaface')

    def _init_pipeline(self, cfg):
        device = get_device()

        def naive_pipeline(image):
            image = np.float32(imread(image))
            image -= (104, 117, 123)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).to(device)
            return dict(inputs=image)

        return naive_pipeline

    def visualize(self, inputs, preds, show=False, vis_thresh=0.8, **kwargs):
        visualization = []
        for image_path, pred in zip(inputs, preds):
            image = imread(image_path)
            self.visualizer.set_image(image)
            for single_label in pred['predictions']:
                score = single_label[4]
                if score < vis_thresh:
                    continue
                text = f'{score:.4f}'

                landms = single_label[5:]
                self.visualizer.draw_bboxes(single_label[:4])
                self.visualizer.draw_texts(text, single_label[:2])
                self.visualizer.draw_circles(landms[:2], np.array([1]),
                                             (0, 0, 225))
                self.visualizer.draw_circles(landms[2:4], np.array([1]),
                                             (0, 255, 225))
                self.visualizer.draw_circles(landms[4:6], np.array([1]),
                                             (255, 0, 225))
                self.visualizer.draw_circles(landms[6:8], np.array([1]),
                                             (0, 255, 0))
                self.visualizer.draw_circles(landms[8:10], np.array([1]),
                                             (255, 0, 0))
            if show:
                self.visualizer.show()
            vis_result = self.visualizer.get_image()
            visualization.append(
                dict(image=vis_result, filename=osp.basename(image_path)))
        return visualization

    def postprocess(self, preds, visualization, *args, **kwargs):
        if self.save_path is not None:
            for vis in visualization:
                image, filename = vis['image'], vis['filename']
                imwrite(image, osp.join(self.save_path, filename))
        return dict(predictions=preds, visualization=visualization)
