#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
import os, sys
import shutil
import numpy as np
import random

def crop_im(im, x, y, w, h):
    M = np.float32([[1, 0, -x], [0, 1, -y]])
    dst =  cv2.warpAffine(im, M, dsize=(w, h), borderValue=(128, 128, 128), flags=cv2.INTER_LINEAR)
    return dst

class SVHN():
    def __init__(self, main_dir, info_fmt=['t', 'b', 'l', 'r', 'lb']):
        self._data_dir = dict()
        self._info = dict()
        self._data_dir['extra'] = os.path.join(main_dir, 'extra')
        self._data_dir['train'] = os.path.join(main_dir, 'train')
        self._data_dir['test']  = os.path.join(main_dir, 'test')
        self._info['extra'] = os.path.join(self._data_dir['extra'], 'extra.txt')
        self._info['train'] = os.path.join(self._data_dir['train'], 'train.txt')
        self._info['test']  = os.path.join(self._data_dir['test'], 'test.txt')
        assert os.path.exists(self._data_dir['extra']) and os.path.exists(self._data_dir['train']) and os.path.exists(self._data_dir['test'])
        assert os.path.exists(self._info['extra']) and os.path.exists(self._info['train']) and os.path.exists(self._info['test'])
        self._fmt = info_fmt

    def info_generator(self, phase='train'):
        assert phase == 'extra' or phase == 'train' or phase == 'test'
        with open(self._info[phase], 'r') as f:
            for line in f.readlines():
                im_path, *info = line.strip('\n').split(' ')
                assert len(info) % len(self._fmt) == 0
                im_info  = []
                for i in range(len(info) // len(self._fmt)):
                    tmp = dict()
                    for key, value in zip(self._fmt, info[i*len(self._fmt):(i+1)*len(self._fmt)]):
                        tmp[key] = int(value)
                    im_info.append(tmp)
                im = cv2.imread(os.path.join(self._data_dir[phase], im_path))
                assert im is not None
                yield im_path, im, im_info

    def aug(self, im, im_info, num=1, expd=0.3, size = 64, crop = 54):
        '''
        expend bbox by expd, then resize to size(w,h), finally crop(w,h)
        '''
        if not isinstance(expd, tuple):
            expd = (expd, expd)
        if not isinstance(size, tuple):
            size = (size, size)
        if not isinstance(crop, tuple):
            crop = (crop, crop)
        assert 'l'  in  im_info[0] and 'r'  in  im_info[0] and 't'  in  im_info[0] and 'b'  in  im_info[0]
        wapper_l = min([item['l'] for item in im_info])
        wapper_r = max([item['r'] for item in im_info])
        wapper_t = min([item['t'] for item in im_info])
        wapper_b = max([item['b'] for item in im_info])

        w = wapper_r - wapper_l
        h = wapper_b - wapper_t
        
        x = int(wapper_l - w*expd[0]/2 + 0.5)
        y = int(wapper_t - h*expd[1]/2 + 0.5)
        w = int((1+expd[0]) * w + 0.5)
        h = int((1+expd[1]) * h + 0.5)

        dst = cv2.resize(crop_im(im, x, y, w, h), dsize=size)
        # cv2.imshow('xxx', dst)
        dst_list = []
        # for i in range(num):
        #     crop_l = random.sample(range(size[0]-crop[0]), 1)[0]
        #     crop_t = random.sample(range(size[1]-crop[1]), 1)[0]
        #     crop_r = crop_l + crop[0]
        #     crop_b = crop_t + crop[1]
        #     dst_list.append(dst[crop_t:crop_b,crop_l:crop_r])
        dst_list.append(dst)
        return dst_list

    def save_aug(self, aug_dir, num=1, phase='train'):
        if os.path.exists(aug_dir):
            shutil.rmtree(aug_dir)
        os.mkdir(aug_dir)
        gen = self.info_generator(phase)
        count = 0
        hist  = [0, 0, 0, 0, 0, 0, 0, 0]
        with open(os.path.join(aug_dir, phase+'.txt'), 'w') as f:
            for im_path, im, im_info in gen:
                ims = self.aug(im, im_info, num)
                im_str = ''.join([str(item['lb']%10) for item in im_info])
                im_name = im_path.split('.')[0]
                for one_im in ims:
                    cv2.imwrite(os.path.join(aug_dir, '{:0>8d}_{}_{}.jpg'.format(count, im_name, im_str)), one_im)
                    line_str = '{:0>8d}_{}_{}.jpg'.format(count, im_name, im_str)
                    for item in im_info:
                        line_str += ' {}'.format(item['lb'] % 10)
                    print(line_str, file=f)
                    count += 1
                    if count % 100 == 0:
                        print(phase, count)
                hist[len(im_info)] += 1
        with open(os.path.join(aug_dir, phase+'_data.txt'), 'w') as f:
            print(hist, file=f)
                                




if __name__ == '__main__':
    # src = cv2.imread('../train/1.png')
    # print(src.shape)
    # cv2.imshow('src1', src)
    # dst = crop_im(src, -src.shape[1]//3, 0, src.shape[1], src.shape[0])
    # cv2.imshow('dst', dst)
    # cv2.waitKey()

    # svhn = SVHN('../')
    # train_walker = svhn.info_generator('train')
    # im, im_info = next(train_walker)
    # cv2.imshow('src', im)
    # print(im.shape, im_info)
    # dst_ims = svhn.aug(im, im_info, 1)
    # cv2.imshow('dst', dst_ims[0])
    # cv2.waitKey()

    svhn = SVHN('../')
    svhn.save_aug('../extra_aug', phase='extra')
    svhn.save_aug('../train_aug', phase='train')
    svhn.save_aug('../test_aug', phase='test')

