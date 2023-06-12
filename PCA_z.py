from scipy import stats
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from options import get_config

def pca(X,k):#k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix 协方差矩阵
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return feature, data



def main():
    opts = get_config()
    # 1.读取数据
    Z = np.load('./result/38/vec_z/fault_' + opts.test_dic + '_z_vec.npy')
    z_gen = np.load('./result/38/vec_z/fault_' + opts.test_dic + '_z_gen_vec.npy')

    dataz = np.load('./result/38/loss/fault_00_Z' + '.npy')
    limit = dataz.tolist()
    limit.sort()
    T_z = np.array(limit)
    t_idx = round(np.shape(dataz)[0] * 0.98)
    taoZ = T_z[t_idx]

    # 2.LDA特征提取
    W, data = pca(Z, 100)  # 得到投影矩阵
    '''选择合适维度的故障子空间'''
    for i in range(100):
        p1 = W[0:i, :]
        E = np.eye(opts.z_size)

        z_new_res = np.dot(Z, (E - np.dot(p1.T, p1)))

        z_out1_res = np.mean(abs(z_new_res - z_gen), axis=-1)
        z_out_res = np.sqrt(z_out1_res)
        z_out_res = np.reshape(z_out_res, [-1])

        m = len(z_out_res)
        a=0
        for j in range(m):
            if z_out_res[j] < taoZ:
                a = a+1

        if 100*a/m > 98:
            print(i)
            np.save('./result/38/fault_direction/PCA/fault_' + opts.test_dic + '_z_direction.npy', p1)
            break

    E = np.eye(opts.z_size)
    p1 = np.load('./result/38/fault_direction/PCA/fault_' + opts.test_dic + '_z_direction.npy')

    z_new_res = np.dot(Z, (E - np.dot(p1.T, p1)))

    z_out1_res = np.mean(abs(z_new_res - z_gen), axis=-1)
    z_out_res = np.sqrt(z_out1_res)
    z_out_res = np.reshape(z_out_res, [-1])

    z_out1 = np.mean(abs(Z - z_gen), axis=-1)
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
    plt.plot(z_out, label="$LSTM-GAN-\mathrm{\\vartheta_{MSE}}\left(x_i\\right)$", color="blue")
    plt.plot(z_out_res, label="$LSTM-GAN-\mathrm{\\^\\vartheta{_{MSE}}}\left(x_i\\right)$", color="black")
    plt.plot(tao_Z, label="Control limit", color="red", linewidth=3)
    plt.plot()
    plt.legend()
    plt.savefig("./result/38/diagnosis/PCA/fault"+opts.test_dic+"-lstm-gan_z.png")
    plt.show()


if '__main__' == __name__:
    main()
