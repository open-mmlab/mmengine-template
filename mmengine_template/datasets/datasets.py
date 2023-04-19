import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mmengine_template.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class WiderFaceDataset(Dataset):

    def __init__(self, annotations, pipeline, training=True):
        self.imgs_path = []
        self.words = []
        f = open(annotations)
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = annotations.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)
        f.close()
        # NOTE Only for debug
        # self.words = self.words[:10]
        # self.imgs_path = self.imgs_path[:10]
        self.pipeline = TRANSFORMS.build(pipeline)
        self.training = training

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        result = dict()
        data_samples = dict()
        result['data_samples'] = data_samples
        img = cv2.imread(self.imgs_path[index])

        labels = self.words[index]
        data_samples['ori_shape'] = np.array(img.shape[:2])
        data_samples['filename'] = self.imgs_path[index]
        if self.training:
            annotations = np.zeros((0, 15))
            if len(labels) == 0:
                return annotations
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 15))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]  # l0_x
                annotation[0, 5] = label[5]  # l0_y
                annotation[0, 6] = label[7]  # l1_x
                annotation[0, 7] = label[8]  # l1_y
                annotation[0, 8] = label[10]  # l2_x
                annotation[0, 9] = label[11]  # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if (annotation[0, 4] < 0):
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

                annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)

            img, target = self.pipeline(img, target)
            img = torch.from_numpy(img)
            result['inputs'] = img
            data_samples['annotations'] = torch.from_numpy(target)
        else:
            img = self.pipeline(img, training=False)
            result['inputs'] = torch.from_numpy(img)

        return result
