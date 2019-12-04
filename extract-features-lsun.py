#!/bin/sh
import io
import os

import lmdb
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

lsun_root = '/mnt/data/datasets/lsun'
with open(os.path.join(lsun_root, 'category_indices.txt'), 'r') as f:
    cats = [line.split()[0] for line in f]
splits = ['train', 'val']


for s in splits:
    m_feat = []
    r_feat = []
    labels = []
    for label, c in enumerate(cats, 1):
        dname = '{}_{}_lmdb'.format(c, s)
        db_path = os.path.join(lsun_root, dname)
        env = lmdb.open(db_path, map_size=1099511627776,
                        max_readers=100, readonly=True)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                img = T(Image.open(io.BytesIO(val))).unsqueeze(0)
                print(key, img.shape)
                with torch.no_grad():
                    x = mnasnet_features(mnasnet, img)
                    m_feat.append(x)
                    x = resnet_features(resnet34, img)
                    r_feat.append(x)

    m_feat = np.concatenate(m_feat)
    r_feat = np.concatenate(r_feat)
    labels = np.array(labels, dtype='uint8')
    np.save('{}_mnasnet'.format(s), m_feat)
    np.save('{}_resnet'.format(s), r_feat)
    np.save('{}_labels'.format(s), labels)
