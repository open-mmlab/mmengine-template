import copy
import random

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmengine_template.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):

    def transform(self, results) -> None:
        filepath = results['filepath']
        results['img'] = mmcv.imread(filepath)
        return results


@TRANSFORMS.register_module()
class LoadAnnotations(BaseTransform):

    def transform(self, results) -> None:
        annotations = copy.deepcopy(results['annotations'])
        h, w = results['img'].shape[:2]
        if len(annotations) == 0:
            return np.zeros((0, 15))
        for annotation_ in annotations:
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = annotation_[0]  # x1
            annotation[0, 1] = annotation_[1]  # y1
            annotation[0, 2] = annotation_[0] + annotation_[2]  # x2
            annotation[0, 3] = annotation_[1] + annotation_[3]  # y2

            # landmarks
            annotation[0, 4] = annotation_[4]  # l0_x
            annotation[0, 5] = annotation_[5]  # l0_y
            annotation[0, 6] = annotation_[7]  # l1_x
            annotation[0, 7] = annotation_[8]  # l1_y
            annotation[0, 8] = annotation_[10]  # l2_x
            annotation[0, 9] = annotation_[11]  # l2_y
            annotation[0, 10] = annotation_[13]  # l3_x
            annotation[0, 11] = annotation_[14]  # l3_y
            annotation[0, 12] = annotation_[16]  # l4_x
            annotation[0, 13] = annotation_[17]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        annotations = np.array(annotations)
        results['annotations'] = annotations
        return results


@TRANSFORMS.register_module(BaseTransform)
class Pad:

    def __init__(self, size, pad_value) -> None:
        assert isinstance(size, (int, tuple, list))
        self.size = size if isinstance(size, (list, tuple)) else [size, size]
        self.pad_value = pad_value

    def transform(self, results):
        if 'crop' in results:
            return results
        img = results['img']
        img = mmcv.impad(img, self.size, pad_val=self.pad_value)
        results['img'] = img
        return results


@TRANSFORMS.register_module(BaseTransform)
class Resize(BaseTransform):

    def __init__(self, size, keep_ratio=True) -> None:
        assert isinstance(size, (int, tuple, list))
        self.size = size if isinstance(size, (list, tuple)) else [size, size]
        self.keep_ratio = keep_ratio

    def transform(self, results):
        img = results['img']
        img, scale_factor = mmcv.imrescale(
            img, self.size, return_scale=True, interpolation='bilinear')
        results['img'] = img
        results['scale_factor'] = scale_factor
        results['annotations'][:, 0::2] *= scale_factor[1]
        results['annotations'][:, 1::2] *= scale_factor[0]
        return results


@TRANSFORMS.register_module(BaseTransform)
class RandomFlip(BaseTransform):

    def __init__(self, flip_ratio=0.5) -> None:
        self.flip_ratio = flip_ratio

    def transform(self, results):
        img = results['img']
        if np.random.rand() > self.flip_ratio:
            img = mmcv.imflip(img)
            _, width, _ = img.shape
            results['img'] = img
            results['flip'] = True
            annotations = results['annotations']

            boxes = annotations[:, :4]
            landms = annotations[:, 4:-1]

            # boxes
            boxes[:, 0::2] = width - boxes[:, 2::-2]

            # landm
            landms = landms.reshape([-1, 5, 2])
            landms[:, :, 0] = width - landms[:, :, 0]
            tmp = landms[:, 1, :].copy()
            landms[:, 1, :] = landms[:, 0, :]
            landms[:, 0, :] = tmp
            tmp1 = landms[:, 4, :].copy()
            landms[:, 4, :] = landms[:, 3, :]
            landms[:, 3, :] = tmp1
            landms = landms.reshape([-1, 10])

            annotations[:, :4] = boxes
            annotations[:, 4:-1] = landms
            results['annotations'] = annotations
        else:
            results['flip'] = False
        return results


@TRANSFORMS.register_module()
class Crop(BaseTransform):

    def __init__(self, crop_scale=[0.3, 0.45, 0.6, 0.8, 1.0]):
        self.crop_scale = crop_scale

    def matrix_iof(self, a, b):
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / np.maximum(area_a[:, np.newaxis], 1)

    def transform(self, results):
        img = results['img']
        height, width, _ = img.shape
        annotations = results['annotations']
        boxes = annotations[:, :4]
        landms = annotations[:, 4:-1]
        labels = annotations[:, -1]

        for _ in range(250):
            """if random.uniform(0, 1) <= 0.2:

            scale = 1.0
            else:
                scale = random.uniform(0.3, 1.0)
            """
            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]

            scale = random.choice(PRE_SCALES)
            short_side = min(width, height)
            width = int(scale * short_side)
            height = width
            if width == width:
                left = 0
            else:
                left = random.randrange(width - width)
            if height == height:
                top = 0
            else:
                top = random.randrange(height - height)
            roi = np.array((left, top, left + width, top + height))

            value = self.matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers,
                                    centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()
            landms_t = landms[mask_a].copy()
            landms_t = landms_t.reshape([-1, 5, 2])

            if boxes_t.shape[0] == 0:
                continue

            image_t = img[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            # landm
            landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
            landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2],
                                            np.array([0, 0]))
            landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2],
                                            roi[2:] - roi[:2])
            landms_t = landms_t.reshape([-1, 10])

            # make sure that the cropped image contains at least one face > 16
            # pixel at training image scale
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / width
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / height
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]
            landms_t = landms_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue

            results['img'] = image_t
            annotations[:, :4] = boxes_t
            annotations[:, 4:-1] = landms_t
            annotations[:, -1] = labels_t
            results['annotations'] = annotations
            results['crop'] = roi
            return results
        return results


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):

    def transform(self, results):

        img = results['img']

        if random.randrange(2):

            # brightness distortion
            if random.randrange(2):
                self.convert(img, beta=random.uniform(-32, 32))

            # contrast distortion
            if random.randrange(2):
                self.convert(img, alpha=random.uniform(0.5, 1.5))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # saturation distortion
            if random.randrange(2):
                self.convert(img[:, :, 1], alpha=random.uniform(0.5, 1.5))

            # hue distortion
            if random.randrange(2):
                tmp = img[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                img[:, :, 0] = tmp

            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        else:

            # brightness distortion
            if random.randrange(2):
                self.convert(img, beta=random.uniform(-32, 32))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # saturation distortion
            if random.randrange(2):
                self.convert(img[:, :, 1], alpha=random.uniform(0.5, 1.5))

            # hue distortion
            if random.randrange(2):
                tmp = img[:, :, 0].astype(int) + random.randint(-18, 18)
                tmp %= 180
                img[:, :, 0] = tmp

            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            # contrast distortion
            if random.randrange(2):
                self.convert(img, alpha=random.uniform(0.5, 1.5))

        results['img'] = img
        return results

    def convert(self, img, alpha=1, beta=0):
        tmp = img.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        img[:] = tmp
