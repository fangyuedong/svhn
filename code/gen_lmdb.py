#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os,sys
import shutil
import caffe
import lmdb
from PIL import Image 
import numpy as np 
import random
import cv2

MAX_LEN = 6

def svhn_lmdb(dir, no_num=False):
    if os.path.exists(os.path.join(dir, 'train.txt')):
        file_path = os.path.join(dir, 'train.txt')
    elif os.path.exists(os.path.join(dir, 'test.txt')):
        file_path = os.path.join(dir, 'test.txt')
    elif os.path.exists(os.path.join(dir, 'extra.txt')):
        file_path = os.path.join(dir, 'extra.txt')
    else:
        raise EnvironmentError

    if os.path.exists(file_path[:-4] + '_im.db'):
        shutil.rmtree(file_path[:-4] + '_im.db')
    if os.path.exists(file_path[:-4] + '_lb.db'):
        shutil.rmtree(file_path[:-4] + '_lb.db')
    im_db = lmdb.open(file_path[:-4] + '_im.db', map_size=int(1e12))
    lb_db = lmdb.open(file_path[:-4] + '_lb.db', map_size=int(1e12))
    im_txn = im_db.begin(write=True)
    lb_txn = lb_db.begin(write=True)
    print(file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        rand_idxs = random.sample(range(len(lines)), len(lines))
        count = 0
        for idx, line in zip(rand_idxs, lines):
            im_path, *labels = line.strip('\n').split(' ')
            if len(im_path) == 0:#skip empty line
                continue
            im = np.array(Image.open(os.path.join(dir, im_path)))
            # cv2.imshow('xxx', im)
            im = im[:,:,::-1].transpose((2,0,1))
            im_data = caffe.io.array_to_datum(im)
            im_txn.put('{:0>10d}'.format(idx).encode(),im_data.SerializeToString())
            # im_txn.commit()

            if no_num == False:
                #ignore label assign to -1
                target = -np.ones((MAX_LEN+1, 1, 1))
                target[-1,0,0] = len(labels) - 1
                for i in range(len(labels)):
                    target[i,0,0] = labels[i]
            else:
                target = 10 * np.ones((MAX_LEN, 1, 1))
                for i in range(len(labels)):
                    target[i,0,0] = labels[i] 
                           
            # print(target)
            lb_data = caffe.io.array_to_datum(target)
            lb_txn.put('{:0>10d}'.format(idx).encode(),lb_data.SerializeToString())
            # lb_txn.commit()
            # cv2.waitKey()
            if count % 100 == 0:
                print(count)
            count += 1
    im_txn.commit()
    lb_txn.commit()
    im_db.close()
    lb_db.close()

if __name__ == '__main__':
    svhn_lmdb('../extra_aug', no_num=True)
    svhn_lmdb('../test_aug', no_num=True)
    svhn_lmdb('../train_aug', no_num=True)

