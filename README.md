# I-LSTM-GAN:动态系统故障诊断模型基于tf.keras实现
---
revised by 杜文友 杨峻培
## 目录

1. 创建环境
2. 准备数据集
3. 配置参数文件options.py
4. 训练与测试
5. 故障子空间提取
6. 故障诊断
---

## a.创建环境
利用requirement.txt创建（python>=3.9）

    pip install -r requirement.txt

---

## b.准备数据集
### 1、训练TEP数据集(训练数据仅采用正常数据集d00.npy)
数据集目录安排如下：


    I-LSTM-GAN
    --data
    ----npy
    ------d00.npy
    ------d00_te.npy
   
### 2、训练自己的数据集
需要自己搭建接口文件，将自己的数据集规范为TEP数据集标准格式，利用参数配置模块(args)进行数据集的输入。
---

## c. 配置参数文件
参数文件主要在options.py文件中

1、选择使用的数据集

2、输入数据集的接口路径

3、更新变量维度信息

4、空行下面的参数可以依据实际情况进行微调或优化，但TEP数据集不建议更改

---

## d. 训练与测试
1、检查好上述的信息后，使用main.py进行训练。权重保存在ckpt文件夹中。

2、测试前在options.py文件中输入需要诊断数据集的路径。然后使用test.py进行测试，得到故障指示量曲线、MAR、FAR。

---

## e. 故障子空间提取
在PCA_z.py文件中

1、输入为故障检测模型中潜空间向量

    Z = np.load('./result/38/vec_z/fault_' + opts.test_dic + '_z_vec.npy')
    z_gen = np.load('./result/38/vec_z/fault_' + opts.test_dic + '_z_gen_vec.npy')
    dataz = np.load('./result/38/loss/fault_00_Z' + '.npy')
    
其中z为故障检测模型中故障数据集下的潜空间向量；z_gen为故障检测模型中逆变器生成的故障数据集下的潜空间向量；dataz为故障检测模型中正常数据集下的潜空间向量
data_z用于构建控制限，z_gen用于故障指示量的计算，Z用于故障子空间的提取

2、运行PCA_z.py,结果展示为故障子空间提取重构前后的故障指示量图和输出显示故障子空间维度

---


## f. 故障诊断

在zhenduan.py文件中

1、在options.py文件中修改test_dic参数，表示为用于诊断的故障类型


    parser.add_argument('--test_dic', help='与故障测试数据集保持一致，便于保存', default='02', type=str)
    

2、运行程序，输出各故障子空间重构故障数据前后的故障指示量图，并输出显示出现的故障类型编号
