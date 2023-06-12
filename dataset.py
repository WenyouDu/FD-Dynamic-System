import numpy as np


def dataload(opts, is_train):

    seq_length = opts.seq_length
    seq_step = opts.seq_step
    if is_train:
        train_ = np.load(opts.train_data)
        print('加载训练集数据 %s' % opts.train_data)
        m_tr, n_tr = train_.shape  #
        train = np.zeros((m_tr, n_tr))
        for i in range(n_tr):
            A = max(train_[:, i])
            a = min(train_[:, i])

            # scale from 0 to 1
            train[:, i] = 2 * ((train_[:, i] - a) / (A - a)) - 1

        samples_train = train[:, 0: n_tr]
        labels_train = np.zeros([m_tr, 1])  # the last colummn is label


        num_samples_train = ((samples_train.shape[0] - seq_length) // seq_step) + 1
        aa = np.empty([num_samples_train, seq_length, n_tr])
        bb = np.empty([num_samples_train, seq_length, 1])

        for j in range(num_samples_train):
            bb[j, :, :] = np.reshape(labels_train[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
            for i in range(n_tr):
                aa[j, :, i] = samples_train[(j * seq_step):(j * seq_step + seq_length), i]

        samples_train = aa
        labels_train = bb

        return samples_train, labels_train

    else:
        if opts.test_dic=='00':
            test_ = np.load(opts.test_data)
        else:
            test_ = (np.load(opts.test_data)).T
        print('加载测试集数据 %s' % opts.test_data)
        test = np.zeros(test_.shape)
        train_ = np.load(opts.train_data)
        m_te, n_te = train_.shape  #
        for i in range(n_te):
            A = max(train_[:, i])
            a = min(train_[:, i])

            # scale from 0 to 1
            test[:, i] = 2 * ((test_[:, i] - a) / (A - a)) - 1

        samples_test = test[:, 0: n_te]
        labels_1 = np.zeros([160, 1])
        labels_2 = np.ones([800, 1])
        labels_test = np.concatenate((labels_1, labels_2), axis=0)

        num_samples_test = ((samples_test.shape[0] - seq_length) // seq_step) + 1
        aa_t = np.empty([num_samples_test, seq_length, n_te])
        bb_t = np.empty([num_samples_test, seq_length, 1])

        for j in range(num_samples_test):
            bb_t[j, :, :] = np.reshape(labels_test[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
            for i in range(n_te):
                aa_t[j, :, i] = samples_test[(j * seq_step):(j * seq_step + seq_length), i]

        samples_test = aa_t
        labels_test = bb_t

        return samples_test, labels_test


def dataload_X(seq_length, seq_step, num_signals, fault_dic):
    train = np.load('./data/npy/d00.npy')
    print('load d00 from .npy')

    test = np.load('./data/npy/d'+ fault_dic +'_te.npy').T
    # test = np.load('./data/npy/d00.npy')
    print('load d01_te from .npy')

    m_tr, n_tr = train.shape  #
    m_te, n_te = test.shape

    for i in range(n_tr):
        A = max(train[:, i])
        a = min(train[:, i])

        # scale from 0 to 1
        train[:, i] = 2*((train[:, i]-a) / (A-a)) - 1

        test[:, i] = 2*((test[:, i]-a) / (A-a)) - 1


    samples_train = train[:, 0: n_tr]
    labels_train = np.zeros([m_tr, 1])  # the last colummn is label

    samples_test = test[:, 0: n_te]
    labels_1 = np.zeros([160, 1])
    labels_2 = np.ones([800, 1])
    labels_test = np.concatenate((labels_1, labels_2), axis=0)
    index = np.asarray(list(range(0, m_te)))  # record the idx of each point

    num_samples_train = ((samples_train.shape[0] - seq_length) // seq_step) +1
    aa = np.empty([num_samples_train, seq_length, n_tr])
    bb = np.empty([num_samples_train, seq_length, 1])

    for j in range(num_samples_train):
        bb[j, :, :] = np.reshape(labels_train[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(n_tr):
            aa[j, :, i] = samples_train[(j * seq_step):(j * seq_step + seq_length), i]

    num_samples_test = ((samples_test.shape[0] - seq_length) // seq_step) + 1
    aa_t = np.empty([num_samples_test, seq_length, n_te])
    bb_t = np.empty([num_samples_test, seq_length, 1])
    bbb_t = np.empty([num_samples_test, seq_length, 1])

    for j in range(num_samples_test):
        bb_t[j, :, :] = np.reshape(labels_test[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        bbb_t[j, :, :] = np.reshape(index[(j * seq_step):(j * seq_step + seq_length)], [-1, 1])
        for i in range(n_te):
            aa_t[j, :, i] = samples_test[(j * seq_step):(j * seq_step + seq_length), i]

    samples_train = aa
    labels_train = bb

    samples_test = aa_t
    labels_test = bb_t
    index = bbb_t
    return samples_train, labels_train, samples_test, labels_test, index


def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    if labels is None:
        return samples[start_pos:end_pos], None
    else:
        if type(labels) == tuple: # two sets of labels
            assert len(labels) == 2
            return samples[start_pos:end_pos], labels[0][start_pos:end_pos], labels[1][start_pos:end_pos]
        else:
            assert type(labels) == np.ndarray
            return samples[start_pos:end_pos], labels[start_pos:end_pos]


def data_slipt(data, seq_length, seq_step):
    m_te, n_te = data.shape
    num_samples_test = ((data.shape[0] - seq_length) // seq_step) + 1
    aa_t = np.empty([num_samples_test, seq_length, n_te])

    for j in range(num_samples_test):
        for i in range(n_te):
            aa_t[j, :, i] = data[(j * seq_step):(j * seq_step + seq_length), i]

    data_s = aa_t

    return data_s