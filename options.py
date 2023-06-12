class Options(object):
  pass

def get_config():
  opts = Options()

  # 训练数据文件选择
  opts.train_data = './data/npy/d00.npy'
  opts.test_data = './data/npy/d04_te.npy'
  opts.test_dic = '04'  #与训练数据同故障名，便于保存文件
  opts.result_dir = "result"

  opts.batch_size = 8
  opts.seq_length = 38
  opts.seq_step = 1
  opts.num_signals = 52

  opts.lr = 1e-4
  opts.iteration = 50
  opts.ckpt_dir = "ckpt"
  opts.z_size = 100
  opts.test_batch_size = 8

  opts.fault_class = 6

  return opts
