#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
import copy
import cv2
import numpy as np
import caffe


def load(prototxt, caffemodel, gpu_id=0):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    return caffe.Net(prototxt, caffemodel, caffe.TEST)

def preprocess(im_path, scale=1, crop=(0,0)):
    im = cv2.imread(im_path)*scale
    if crop == (0,0):
        crop_im = im
    else:
        h,w = im.shape[0:2]
        crop_im = im[(h-crop[0])//2:(h-crop[0])//2+crop[0],(w-crop[1])//2:(w-crop[1])//2+crop[1],:]
    return crop_im[:,:,::-1].transpose(2,0,1)

def forward(net, im_paths, scale=1, crop=(0,0)):
    n,c,h,w = net.blobs['data'].shape
    for i in range((len(im_paths)+n-1)//n):
        ims = [preprocess(im_path,scale,crop) for im_path in im_paths[i*n:min((i+1)*n,len(im_paths))]]
        net.blobs['data'].data[:min(n,len(im_paths)-i*n),:,:,:] = np.array(ims)
        batch_out = net.forward()
        if i == 0:
            output = copy.deepcopy(batch_out)
        else:
            for layer_name, _ in batch_out.items():
                output[layer_name] = np.concatenate((output[layer_name], batch_out[layer_name]),axis=0)
            print(i)
    return output


if __name__ == "__main__":
    net = load("half_res18_non_svhn2.0_deploy.prototxt",
        "../model/half_res18_non_svhn2.0_iter_17000.caffemodel",
        0)
    # for layer_name, layer in net.blobs.items():
    #     print(layer_name, layer.data.shape)
    res = forward(net, ["../test_aug/00000001.jpg"], 1/255, crop=(54,54))
    for layer_name, layer in res.items():
        print(layer_name, layer.data.shape)
    # im = cv2.imread("../test_aug/00000001.jpg")
    # cv2.imshow("im", im)
    # crop_im = im[20:-20,5:-5,:]
    # cv2.imshow("crop_im", crop_im[:,:,::-1])
    # cv2.waitKey()
    # print(crop_im.shape)

