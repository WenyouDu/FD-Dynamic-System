from options import get_config
import matplotlib.pyplot as plt
import numpy as np


opts = get_config()
# 1.读取数据
Z_f = np.load('./result/38/vec_z/fault_' + opts.test_dic + '_z_vec.npy')
z_gen = np.load('./result/38/vec_z/fault_' + opts.test_dic + '_z_gen_vec.npy')
Z0 = np.load('./result/38/vec_z/fault_00_z_vec.npy')
m_f, n_f = Z0.shape

# 2.各个故障子空间重构故障数据
for i in range(opts.fault_class):
    p1 = np.load('./result/38/fault_direction/PCA/fault_' + str(i+1).zfill(2) +'_z_direction.npy')
    P = np.load('./result/38/fault_direction/PCA/fault_01_z_direction.npy')
    E = np.eye(opts.z_size)

    z_new_res = np.dot(Z_f, (E - np.dot(p1.T, p1)))

    z_out1_res = np.mean(abs(z_new_res - z_gen), axis=-1)
    z_out_res = np.sqrt(z_out1_res)
    z_out_res = np.reshape(z_out_res, [-1])

    z_out1 = np.mean(abs(Z_f - z_gen), axis=-1)
    z_out = np.sqrt(z_out1)
    z_out = np.reshape(z_out, [-1])

    dataz = np.load('./result/38/loss/fault_00_Z' + '.npy')
    Z = dataz.tolist()
    Z.sort()
    T_z = np.array(Z)
    t_idx = round(np.shape(dataz)[0] * 0.98)
    tao_Z = T_z[t_idx] * np.ones(z_out.shape)
    taoZ = T_z[t_idx]

    plt.xlabel("Samples")
    plt.ylabel("Statistics")
    plt.plot(z_out, label="LSTM-GAN_Z", color="blue", marker='.')
    plt.plot(z_out_res, label="LSTM-GAN_reconstruction", color="black", marker='.')
    plt.plot(tao_Z, label="Control_limit_z", color="red", linewidth=3, linestyle='dashed')
    plt.plot()
    plt.legend()
    plt.savefig("./result/38/diagnosis/fault" + opts.test_dic + '_' + str(i+1)+"-lstm-gan.png")
    plt.show()

    m= len(z_out_res)
    a=0
    for j in range(m):
        if z_out_res[j]<taoZ:
            a = a+1

    if 100*a/m >= 98:
        print(i+1)
        break


