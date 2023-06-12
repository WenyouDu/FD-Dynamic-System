# -*- coding: utf-8 -*-
"""
Created on 2023/5/26

@author: YANG

This a template code for the implement of GAN based on tensorflow 2.8.0

test dataset is TEP fault dataset. train dataset is TEP normal dataset

"""

import os
os.environ['TF_ENABLE_COND_V2'] = '1'
import numpy as np
from ganormal import Ganormal
from options import get_config
from tqdm import tqdm
from dataset import dataload, get_batch

import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR')
tf.disable_v2_behavior()


if __name__ == "__main__":
    ''' 0. prepare data '''

    ''' training takes only normal dataset, and every dataset through serialization processing
        resize the samples into 8*40*52 size in order to fit the model input
    '''

    ''' 1. train model '''
    sess = tf.Session()
    opts = get_config()
    model = Ganormal(sess, opts)

    x_train, _ = dataload(is_train=True, opts=opts)
    ''' 
    strat training
    '''    
    auc_all = []
    for i in range(opts.iteration):
        loss_train_all = []
        loss_test_all = []
        real_losses = []
        fake_losses = []
        enco_losses = []
        ''' shuffle data in each epoch'''
        # permutated_indexes = np.random.permutation(x_train.shape[0])
        ''' decay the learning rate. we dont do that in tensorflow way because it
        is more easier to fine-tuning'''
        for batch_idx in tqdm(range(int(x_train.shape[0] / opts.batch_size))):
            batch_x, batch_y = get_batch(x_train, opts.batch_size, batch_idx)

            z, loss, al, cl, el = model.train(batch_x)
            # print(loss, al, cl, el)
            loss_train_all.append(loss)
            real_losses.append(al)
            fake_losses.append(cl)
            enco_losses.append(el)
        print("iter {:>6d} :{:.4f} a:{:.4f} c {:.4f} e{:.4f}".format(i+1, np.mean(loss_train_all),
                                                  np.mean(real_losses),
                                                  np.mean(fake_losses),
                                                  np.mean(enco_losses)))
        '''save the model'''
        if (i+1) % 10 ==0:
            model.save(opts.ckpt_dir)

