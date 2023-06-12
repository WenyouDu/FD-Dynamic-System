import os
os.environ['TF_ENABLE_COND_V2'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import math

from dataset import dataload, get_batch
from keras.models import load_model
from options import get_config
from ganormal import Ganormal

import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR')
tf.disable_v2_behavior()


class Predict(object):

    def __init__(self):
        # 清除默认图的堆栈，并设置全局图为默认图
        # 若不进行清楚则在第二次加载的时候报错，因为相当于重新加载了两次
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.net = Ganormal(self.sess, opts)
        # 加载模型到sess中
        self.restore()
        print('load susess')

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(opts.ckpt_dir)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('未保存模型')

    def evaluate(self, whole_x, whole_y):
        bs = self.net.opts.test_batch_size

        batch_idx = opts.batch_size*int(whole_x.shape[0] / bs)
        vec_z = np.empty([batch_idx, opts.z_size])
        vec_z_gen = np.empty([batch_idx, opts.z_size])
        z_out = []
        R_out = []
        labels_out = []

        for index in range(int(whole_x.shape[0] / bs)):
            start_pos = index*bs
            end_pos = (index+1)*bs
            batch_x = whole_x[index*bs:(index+1)*bs]
            batch_y = whole_y[index*bs:(index+1)*bs]
            latent_loss, latent_gen_loss = self.sess.run([self.net.latent_z,
                                                          self.net.latent_z_gen],
                                 {self.net.data_input:batch_x, self.net.is_train: False})
            sample_gen_loss = self.sess.run(self.net.data_gen,
                                 {self.net.data_input:batch_x, self.net.is_train: False})

            latent_error1 = np.mean(abs(latent_loss - latent_gen_loss), axis=-1)
            latent_error = np.sqrt(latent_error1)
            latent_error = np.reshape(latent_error, [-1])

            sample_error1 = np.mean(abs(batch_x - sample_gen_loss), axis=-1)
            # SPE = np.dot(sample_error1.T, sample_error1)
            sample_error = np.mean(sample_error1, axis=-1)
            sample_error = np.reshape(sample_error, [-1])

            vec_z[start_pos:end_pos, :] = latent_loss
            vec_z_gen[start_pos:end_pos, :] = latent_gen_loss
            z_out = np.append(z_out, latent_error)
            R_out = np.append(R_out, sample_error)

            labels_out = 10*z_out + R_out
            ''' Scale scores vector between [0, 1]'''
        # z_out = (z_out - z_out.min())/(z_out.max()-z_out.min())
        # R_out = (R_out - R_out.min())/(R_out.max()-R_out.min())

        # labels_out = z_out + R_out

        return z_out, R_out, labels_out, vec_z, vec_z_gen

if __name__ == '__main__':
    opts = get_config()

    x_test, y_test = dataload(opts=opts, is_train=False)

    # sess = tf.Session()
    model = Predict()
    # model = load_model('opts.ckpt_dir')
    z_out, R_out, labels_out, vec_z, vec_z_gen = model.evaluate(x_test, y_test)

    np.save('./result/38/loss/fault_' + opts.test_dic + '_Z.npy', z_out)
    np.save('./result/38/loss/fault_' + opts.test_dic + '_R.npy', R_out)
    np.save('./result/38/loss/fault_' + opts.test_dic + '_Z+R.npy', labels_out)
    np.save('./result/38/vec_z/fault_' + opts.test_dic + '_z_vec.npy', vec_z)
    np.save('./result/38/vec_z/fault_' + opts.test_dic + '_z_gen_vec.npy', vec_z_gen)

    dataz = np.load('./result/38/loss/fault_00_Z' + '.npy')
    Z = dataz.tolist()
    Z.sort()
    T_z = np.array(Z)
    t_idx = round(np.shape(dataz)[0] * 0.98)
    tao_Z = T_z[t_idx] * np.ones(z_out.shape)
    taoZ = T_z[t_idx]

    plt.xlabel("Samples")
    plt.ylabel("Statistics")
    plt.plot(z_out, label="LSTM-GAN_Z", color="blue")
    plt.plot(tao_Z, label="Control_limit_z", color="red", linewidth=3)
    plt.plot()
    plt.legend()
    plt.savefig("./result/38/plots/" + str(opts.seq_length) + 'RT_z_fault' +opts.test_dic+ "T-lstm-gan.png")
    plt.show()


    datar = np.load('./result/38/loss/fault_00_R' + '.npy')
    R = datar.tolist()
    R.sort()
    T_R = np.array(R)
    t_idx = round(np.shape(datar)[0] * 0.98)
    tao_R = T_R[t_idx] * np.ones(z_out.shape)
    taoR = T_R[t_idx]

    plt.xlabel("Samples")
    plt.ylabel("Statistics")
    plt.plot(R_out, label="$LSTM-GAN-\mathrm{\\vartheta_{MAD}}\left(x_i\\right)$", color="blue")
    plt.plot(tao_R, label="Control_limit", color="red", linewidth=3)
    plt.plot()
    plt.legend()
    plt.savefig("./result/38/plots/" + str(opts.seq_length) + 'RT_R_fault'+opts.test_dic+"T-lstm-gan.png")
    plt.show()

    LL = np.shape(z_out)[0]
    index_lable = 160 - (opts.seq_length)
    labels_1 = np.zeros([index_lable, 1])
    labels_2 = np.ones([LL - index_lable, 1])
    L_L = np.concatenate((labels_1, labels_2), axis=0)

    '''performance evaluation index calculation'''
    TP, TN, FP, FN = 0, 0, 0, 0
    Z_pre = np.zeros([LL, 1])
    for i in range(LL):
        if z_out[i] > taoZ:
            # true/negative
            Z_pre[i] = 1
        else:
            # false/positive
            Z_pre[i] = 0

        A = Z_pre[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1
        # confusion_matrix.plt_confusion_matrix(D_L, L_L)

    MAR = (100 * FN) / (TP + FN + 1)
    FAR = (100 * FP) / (FP + TN + 1)
    print('Z : MAR: {:.4}; FAR: {:.4}'.format(MAR, FAR))

    TP1, TN1, FP1, FN1 = 0, 0, 0, 0
    R_pre = np.zeros([LL, 1])
    for i in range(LL):
        if R_out[i] > taoR:
            # true/negative
            R_pre[i] = 1
        else:
            # false/positive
            R_pre[i] = 0

        C = R_pre[i]
        D = L_L[i]
        if C == 1 and D == 1:
            TP1 += 1
        elif C == 1 and D == 0:
            FP1 += 1
        elif C == 0 and D == 0:
            TN1 += 1
        elif C == 0 and D == 1:
            FN1 += 1
        # confusion_matrix.plt_confusion_matrix(D_L, L_L)

    MAR1 = (100 * FN1) / (TP1 + FN1 + 1)
    FAR1 = (100 * FP1) / (FP1 + TN1 + 1)
    print('R : MAR: {:.4}; FAR: {:.4}'.format(MAR1, FAR1))

    ########## Comprehensive ########
    R = R_pre
    Z = Z_pre
    TP2, TN2, FP2, FN2 = 0, 0, 0, 0
    Com_pre = np.zeros([LL, 1])
    for i in range(LL):
        if R[i] == 1 or Z[i] == 1:
            # true/negative
            Com_pre[i] = 1
        else:
            # false/positive
            Com_pre[i] = 0

        E = Com_pre[i]
        F = L_L[i]
        if E == 1 and F == 1:
            TP2 += 1
        elif E == 1 and F == 0:
            FP2 += 1
        elif E == 0 and F == 0:
            TN2 += 1
        elif E == 0 and F == 1:
            FN2 += 1
        # confusion_matrix.plt_confusion_matrix(D_L, L_L)

    Accu = (100 * (TP2 + TN2)) / (TP2 + TN2 + FP2 + FN2)
    Rec = (100 * TP2) / (TP2 + FN2 + 1)
    MAR2 = (100 * FN2) / (TP2 + FN2 + 1)
    FAR2 = (100 * FP2) / (FP2 + TN2 + 1)
    print('C : MAR: {:.4}; FAR: {:.4}; Accu: {:.4}'.format(MAR2, FAR2, Accu))





