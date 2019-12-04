#!/bin/sh
import glob
import os

import numpy as np
import torch
from torchvision import models


mnasnet = models.mnasnet1_0(pretrained=True)
mnasnet.eval()

#vgg16 = models.vgg16_bn(pretrained=True)
#vgg16.eval()

resnet34 = models.resnet34(pretrained=True)
resnet34.eval()

from torchvision import transforms


T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

def mnasnet_features(model, img):
    x = model.layers(img)
    return x.mean([2, 3]).cpu().numpy().astype('float32')

def vgg_features(model, img):
    x = model.features(img)
    x = model.avgpool(x)
    return torch.flatten(x, 1).cpu().numpy().astype('float32')


def resnet_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x.cpu().numpy().astype('float32')


from PIL import Image

voc_root = '/home/darwin/Projects/EE298-DOE/data/VOCdevkit/VOC2012'
ann_root = os.path.join(voc_root, 'ImageSets', 'Main')
cats = sorted({c.split('_')[0] for c in glob.glob(os.path.join(ann_root, '*_*.txt'))})
splits = ['train', 'val']

for s in splits:
    m_feat = []
    r_feat = []
    labels = []
    for label, c in enumerate(cats, 1):
        fname = '{}_{}.txt'.format(c, s)
        with open(os.path.join(ann_root, fname)) as f:
            for line in f:
                iname, included = line.split()
                if included == '1':
                    labels.append(label)
                    p = os.path.join(voc_root, 'JPEGImages', iname + '.jpg')
                    print(label, fname, p)
                    img = T(Image.open(p)).unsqueeze(0)
                    with torch.no_grad():
                        x = mnasnet_features(mnasnet, img)
                        m_feat.append(x)
                        x = resnet_features(resnet34, img)
                        r_feat.append(x)


    m_feat = np.concatenate(m_feat)
    r_feat = np.concatenate(v_feat)
    labels = np.array(labels, dtype='uint8')
    np.save('{}_mnasnet'.format(s), m_feat)
    np.save('{}_resnet'.format(s), r_feat)
    np.save('{}_labels'.format(s), labels)
