# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:22:18 2018

@author: zhang
"""

from __future__ import print_function
from tensorflow.python.keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten, LSTM
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import Activation
# from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers

import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR')
tf.disable_v2_behavior()

import numpy as np
import math
from options import get_config
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


''' type-1 loss and type2 loss'''
def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def l2_loss(y_true, y_pred):
    return K.mean(K.sqrt(y_pred - y_true))


def bce_loss(y_pred,y_true):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                  logits=y_pred))


'''Tensorflow based bat normalizatioin'''
def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
  return tf.layers.batch_normalization(x,
                                      momentum=momentum,
                                      # updates_collections=None,
                                      epsilon=epsilon,
                                      scale=True,
                                      training=is_train,
                                      name=name)


def Encoder(inputs, opts, istrain=True, name='e1'):
    lstm1 = LSTM(100, return_sequences=True)(inputs)   #input_shape=(nb_time_steps, nb_input_vector),
    lstm1 = LeakyReLU(0.2)(lstm1)

    lstm2 = LSTM(200, return_sequences=True)(lstm1)
    lstm2 = LeakyReLU(0.2)(lstm2)

    lstm3 = LSTM(300)(lstm2)
    lstm3 = LeakyReLU(0.2)(lstm3)

    bn1 = batch_norm(lstm3, name + "_bn2", is_train=istrain)
    bn1 = Dropout(0.6)(bn1)
    bn1 = LeakyReLU(0.2)(bn1)

    ''' final layer, resize the layer to batch_size X 100'''

    output = Dense(opts.z_size, activation='tanh')(bn1)
            
    return output


def Decoder(inputs, opts, istrain=True):
    '''z is input, and first deconvolution layer to size channel * 4 * 4'''
    x = Dense(opts.seq_length)(inputs)
    x = tf.reshape(x, [opts.batch_size, opts.seq_length, 1])

    x = batch_norm(x, "bn1", is_train=istrain)
    x = Dropout(0.6)(x)
    x = LeakyReLU(0.2)(x)

    lstm1 = LSTM(100, return_sequences=True)(x)     #input_shape=(nb_time_steps, nb_input_vector),
    lstm1 = LeakyReLU(0.2)(lstm1)

    lstm2 = LSTM(200, return_sequences=True)(lstm1)
    lstm2 = LeakyReLU(0.2)(lstm2)

    lstm3 = LSTM(100, return_sequences=True)(lstm2)
    lstm3 = LeakyReLU(0.2)(lstm3)

    lstm4 = LSTM(opts.num_signals, return_sequences=True)(lstm3)
    ''' final layer, expand the size with 2 and channel of n_output_channel'''
    x = Activation('tanh')(lstm4)
    return x


'''generator is the encoder->decoder->encoder structure'''
def generator(inputs, opts, istrain=True):

    with tf.variable_scope('gen_'):
        # gen_data = Decoder(z, opts, istrain=istrain)
        z      = Encoder(inputs, opts, istrain=istrain, name='e1')
        x_star = Decoder(z, opts, istrain=istrain)
        z_star = Encoder(x_star, opts, istrain=istrain, name='e2')
    return x_star, z, z_star


# def generator(z, opts, istrain=True):
#     gen_data = Decoder(z, opts, istrain=istrain)




''' discriminator is the same as discriminator from DCGan
    the last layer is utilized for feature mapping loss
    in the original code, the dis net is the same as the encoder without final layer
'''

def discriminator(inputs, opts, reuse=False, istrain=True, name='d1'):
    with tf.variable_scope('dis_', reuse=reuse):
        ''' initial layer'''
        lstm1 = LSTM(100, return_sequences=True)(inputs)  # input_shape=(nb_time_steps, nb_input_vector),
        lstm1 = LeakyReLU(0.2)(lstm1)

        lstm2 = LSTM(200, return_sequences=True)(lstm1)
        lstm2 = LeakyReLU(0.2)(lstm2)

        lstm3 = LSTM(100)(lstm2)
        lstm3 = LeakyReLU(0.2)(lstm3)

        x = batch_norm(lstm3, name + "_bn2", is_train=istrain)
        x = Dropout(0.6)(x)
        x = LeakyReLU(0.2)(x)

        feature = x
        # state size. channel x 4 x 4        
#        ''' final layer, resize the layer to channel X 1 X 1'''
        classifier = Dense(opts.seq_length, activation='sigmoid')(x)

        return feature, classifier


'''discriminator and generator together class'''


class Ganormal(object):
    def __init__(self, sess, opts):
       self.sess = sess
       self.is_train = tf.placeholder(tf.bool)
       self.data_shape = [opts.batch_size, opts.seq_length, opts.num_signals]
       self.data_input = tf.placeholder(tf.float32, self.data_shape)
       self.opts = opts

       ''' 0 create model'''   
       with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
           self.data_gen, self.latent_z, self.latent_z_gen = generator(self.data_input, self.opts, self.is_train)
           self.feature_fake, self.label_fake = discriminator(self.data_gen, self.opts, False, self.is_train)
           self.feature_real, self.label_real = discriminator(self.data_input, self.opts, True, self.is_train)
       self.t_vars = tf.trainable_variables()
       self.d_vars = [var for var in self.t_vars if 'dis_' in var.name]
       self.g_vars = [var for var in self.t_vars if 'gen_' in var.name]

       '''1 create losses'''
       self.adv_loss = l2_loss(self.feature_fake, self.feature_real )
#       self.adv_loss = bce_loss(self.label_fake, tf.ones_like(self.label_fake))
       self.context_loss = l1_loss(self.data_input, self.data_gen)
       self.encoder_loss = l2_loss(self.latent_z, self.latent_z_gen)
       self.generator_loss = 1*self.adv_loss + 1*self.context_loss + 1*self.encoder_loss

       '''dis loss: real label reach to 1 and fake label reach to 0'''
       self.real_loss = bce_loss(self.label_real, tf.ones_like(self.label_real)) # real reach to 1
       self.fake_loss = bce_loss(self.label_fake, tf.zeros_like(self.label_fake))
       self.feature_loss = self.real_loss + self.fake_loss #-l2_loss(self.feature_fake, self.feature_real)#
       self.discriminator_loss = self.feature_loss

       '''2 optimize the loss, learning rate and beta1 is from Original code of Pytorch '''
       update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
       with tf.control_dependencies(update_ops):
           with tf.variable_scope(tf.get_variable_scope(), reuse=None):
               self.gen_train_op = tf.train.GradientDescentOptimizer(
                   learning_rate=0.02).minimize(self.generator_loss, var_list=self.g_vars)
               self.dis_train_op = tf.train.GradientDescentOptimizer(
                   learning_rate=0.02).minimize(self.discriminator_loss, var_list=self.d_vars)
       '''3 save the model '''    
       self.saver = tf.train.Saver()
       '''4 initialization'''
       self.sess.run(tf.global_variables_initializer())


    ''' generator training in keras style'''
    def gen_fit(self, batch_x):
        _, z, loss, al, cl, el = self.sess.run([self.gen_train_op,
                                             self.latent_z,
                                             self.generator_loss,
                                             self.adv_loss,
                                             self.context_loss,
                                             self.encoder_loss],
                                             {self.data_input: batch_x, self.is_train: False})
        return z, loss, al, cl, el
    
    ''' discriminator training in keras style'''
    def dis_fit(self, batch_x):     
        _, loss, dis_real_loss, dis_fake_loss = self.sess.run([self.dis_train_op, self.discriminator_loss,
                                  self.real_loss,
                                  self.fake_loss], 
          {self.data_input:batch_x, self.is_train: False})
        return loss, dis_real_loss, dis_fake_loss

    ''' train the model in dis and gen'''
    def train(self, batch_x):
        z, gen_loss, al, cl, el = self.gen_fit(batch_x)
        _, dis_real_loss, dis_fake_loss = self.dis_fit(batch_x)
        # If D loss is zero, then re-initialize netD
        if dis_real_loss < 1e-5 or dis_fake_loss < 1e-5:    
            init_op = tf.initialize_variables(self.d_vars)
            self.sess.run(init_op)
#            print('reinitialize')
        return z, gen_loss, al, dis_real_loss, dis_fake_loss


    def evaluate(self, whole_x, whole_y):
        bs = self.opts.test_batch_size

        z_out = []
        R_out = []

        for index in range(int(whole_x.shape[0] / bs)):
            # start_pos = index*bs
            # end_pos = (index+1)*bs
            batch_x = whole_x[index*bs:(index+1)*bs]
            batch_y = whole_y[index*bs:(index+1)*bs]
            latent_loss, latent_gen_loss = self.sess.run([self.latent_z, 
                                                          self.latent_z_gen], 
                                 {self.data_input:batch_x, self.is_train: False})
            sample_gen_loss = self.sess.run(self.data_gen,
                                 {self.data_input:batch_x, self.is_train: False})

            latent_error = np.mean(abs(batch_x - sample_gen_loss), axis=-1)
            latent_error = np.reshape(latent_error, [-1])

            sampler_error = np.mean(abs(latent_loss - latent_gen_loss), axis=-1)
            sampler_error = np.reshape(sampler_error, [-1])

            z_out = np.append(z_out, latent_error)
            R_out = np.append(R_out, sampler_error)

            # labels_out = np.append(labels_out, batch_y)
            ''' Scale scores vector between [0, 1]'''
            z_out = (z_out - z_out.min())/(z_out.max()-z_out.min())
            R_out = (R_out - R_out.min())/(R_out.max()-R_out.min())

        '''calculate the roc value'''
        # auc_out = roc(labels_out, z_out)
        return z_out, R_out

    def save(self, dir_path):
        self.saver.save(self.sess, dir_path+"/model.ckpt")

