import numpy as np
import matplotlib.pyplot as plt
from options import get_config

opts = get_config(is_train=True)

z_normal = np.load('./result/48/loss/fault_00_Z' + '.npy')
Z = z_normal.tolist()
Z.sort()
T_z = np.array(Z)
z_idx = round(np.shape(z_normal)[0] * 0.98)
taoZ = T_z[z_idx]

r_normal = np.load('./result/48/loss/fault_00_R' + '.npy')
R = r_normal.tolist()
R.sort()
T_r = np.array(R)
r_idx = round(np.shape(z_normal)[0] * 0.98)
taoR = T_r[r_idx]

Z_out = np.load('./result/48/loss/fault_'+ opts.test_dic + '_Z.npy')
R_out = np.load('./result/48/loss/fault_'+ opts.test_dic + '_R.npy')

##########   TP, TN, FP, FN   #######

index_lable = 160 - (opts.seq_length)

LL = np.shape(Z_out)[0]
labels_1 = np.zeros([index_lable, 1])
labels_2 = np.ones([LL - index_lable, 1])
L_L = np.concatenate((labels_1, labels_2), axis=0)

TP, TN, FP, FN = 0, 0, 0, 0
Z_pre = np.zeros([LL, 1])
for i in range(LL):
    if Z_out[i] > taoZ:
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

MAR = (100 * FN) / (TP + FN+1)
FAR = (100 * FP) / (FP + TN+1)
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

MAR1 = (100 * FN1) / (TP1 + FN1+1)
FAR1 = (100 * FP1) / (FP1 + TN1+1)
print('R : MAR: {:.4}; FAR: {:.4}'.format(MAR1, FAR1))


########## Comprehensive ########
R = R_pre
Z = Z_pre
TP2, TN2, FP2, FN2 = 0, 0, 0, 0
Com_pre = np.zeros([LL, 1])
for i in range(LL):
    if R[i] == 0 and Z[i] == 0:
        # true/negative
        Com_pre[i] = 0
    else:
        # false/positive
        Com_pre[i] = 1

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

Accu = (100*(TP2+TN2)) / (TP2+TN2+FP2+FN2)
Rec = (100 * TP2) / (TP2 + FN2 + 1)
MAR2 = (100 * FN2) / (TP2 + FN2+1)
FAR2 = (100 * FP2) / (FP2 + TN2+1)
print('C : MAR: {:.4}; FAR: {:.4}; Accu: {:.4}'.format(MAR2, FAR2, Accu))

