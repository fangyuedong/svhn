#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
import numpy as np
import cv2
from lib.libcaffe import inference as cf


def read_data(dir):
    '''return {'xxx.jpg', '123'}'''
    samples = {}
    with open(os.path.join(dir, dir.split('/')[-1].split('_')[0] + '.txt'),'r') as f:
        lines = f.readlines()
        for line in lines:
            im_path, *labels = line.strip('\n').split(' ')
            if len(im_path) == 0:
                continue
            samples[os.path.join(dir, im_path)] = ''.join([str(lb) for lb in labels])
    return samples

def test(dir, prototxt, caffemodel):
    samples  = read_data(dir)
    im_paths = [path for path, _ in samples.items()]
    labels   = [label for _, label in samples.items()]
    net = cf.load(prototxt, caffemodel, 0)
    output = cf.forward(net, im_paths, 1.0/255, crop=(54,54))

    digit0 = np.argmax(output['digit0_prob'],axis=1)
    digit1 = np.argmax(output['digit1_prob'],axis=1)
    digit2 = np.argmax(output['digit2_prob'],axis=1)
    digit3 = np.argmax(output['digit3_prob'],axis=1)
    digit4 = np.argmax(output['digit4_prob'],axis=1)
    digit5 = np.argmax(output['digit5_prob'],axis=1)
    def digits(i):
        return digit0[i], digit1[i], digit2[i], digit3[i], digit4[i], digit5[i]
    
    if 'digit_num_prob' in output: #output digit0, digit1, ... digit5, digit_num
        digit_num = np.argmax(output['digit_num_prob'],axis=1)+1
    else:
        digit_num = [0 for i in range(len(digit0))]
        for i in range(len(digit0)):
            # digit_num[i] = np.where([digit0[i], digit1[i], digit2[i], digit3[i], digit4[i], digit5[i], 10] == 10)[0]
            digit_num[i] = np.where(np.array(digits(i)+(10,))==10)[0][0]
            # print(digit_num[i])

    count = 0
    for i in range(len(im_paths)):
        # if labels[i] != "{}{}{}{}{}{}".format(digit0[i], digit1[i], digit2[i], digit3[i], digit4[i], digit5[i])[:digit_num[i]]:
        if labels[i] != ''.join([str(digit) for digit in digits(i)])[:digit_num[i]]:
            count += 1
            im = cv2.imread(im_paths[i])
            cv2.imshow(im_paths[i]+"_"+labels[i]+"_"+''.join([str(digit) for digit in digits(i)])[:digit_num[i]], im)
            cv2.waitKey(0)
    print(count)
    
if __name__ == "__main__":
    test("../train_aug", "half_res18_non_svhn2.0_deploy.prototxt", "../model/half_res18_non_svhn2.0_iter_18500.caffemodel")

# if __name__ == "__main__":
#     samples  = read_data("../test_aug")
#     im_paths = [path for path, _ in samples.items()]
#     labels   = [label for _, label in samples.items()]
#     net = cf.load("half_res18_svhn2.0_deploy.prototxt",
#         "../model/half_res18_svhn2.0_iter_21500.caffemodel",
#         0)
#     output = cf.forward(net, im_paths, 1/255, crop=(54,54))
#     digit_num = np.argmax(output['digit_num_prob'],axis=1)+1
#     digit0 = np.argmax(output['digit0_prob'],axis=1)
#     digit1 = np.argmax(output['digit1_prob'],axis=1)
#     digit2 = np.argmax(output['digit2_prob'],axis=1)
#     digit3 = np.argmax(output['digit3_prob'],axis=1)
#     digit4 = np.argmax(output['digit4_prob'],axis=1)
#     digit5 = np.argmax(output['digit5_prob'],axis=1)
#     count = 0
#     for i in range(len(im_paths)):
#         if labels[i] != "{}{}{}{}{}{}".format(digit0[i], digit1[i], digit2[i], digit3[i], digit4[i], digit5[i])[:digit_num[i]]:
#             count += 1
#             if len(labels[i]) == 4:
#                 im = cv2.imread(im_paths[i])
#                 cv2.imshow(labels[i]+"_"+"{}{}{}{}{}{}".format(digit0[i], digit1[i], digit2[i], digit3[i], digit4[i], digit5[i])[:digit_num[i]], im)
#                 cv2.waitKey(0)