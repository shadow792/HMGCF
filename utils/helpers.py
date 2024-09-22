
import numpy as np
import random

def sampling(sampling_mode, train_rate, gt):

    train_rand_idx = []
    gt_1d = np.reshape(gt, [-1])

    if sampling_mode == 'random':

        idx = np.where(gt_1d < 3)[-1]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]
        rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_rate).astype('int32'))
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)

        train_rand_idx = np.array(train_rand_idx)
        train_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_index.append(a[j])
        train_index = np.array(train_index)

        train_index = set(train_index)
        all_index = [i for i in range(len(gt_1d))]
        all_index = set(all_index)

        background_idx = np.where(gt_1d == 0)[-1]
        background_idx = set(background_idx)
        test_index = all_index - train_index - background_idx

        val_count = int(0.01 * (len(test_index) + len(train_index)))
        #val_index = random.sample(test_index, val_count)
        val_index = random.sample(list(test_index), val_count)
        val_index = set(val_index)
        test_index = test_index - val_index

        test_index = list(test_index)
        train_index = list(train_index)
        val_index = list(val_index)

    return train_index, val_index, test_index

def one_hot(gt_mask, height, width):
    gt_one_hot = []
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            temp = np.zeros(2, dtype=np.float32)
            if gt_mask[i, j] == 1:  # Change
                temp[0] = 1
            elif gt_mask[i, j] == 2:  # Unchange
                temp[1] = 1
            gt_one_hot.append(temp)
    gt_one_hot = np.reshape(gt_one_hot, [height, width, 2])
    return gt_one_hot

def make_mask(gt_mask, height, width):
    label_mask = np.zeros([height * width, 2])
    temp_ones = np.ones([2])
    gt_mask_1d = np.reshape(gt_mask, [height * width])
    for i in range(height * width):
        if gt_mask_1d[i] in [1, 2]:  # Only for change and unchange classes
            label_mask[i] = temp_ones
    label_mask = np.reshape(label_mask, [height * width, 2])
    return label_mask

def get_mask_onehot(gt, index):
    height, width = gt.shape
    gt_1d = np.reshape(gt, [-1])
    gt_mask = np.zeros_like(gt_1d)
    for i in range(len(index)):
        gt_mask[index[i]] = gt_1d[index[i]]
        pass
    gt_mask = np.reshape(gt_mask, [height, width])
    sampling_gt = gt_mask
    gt_onehot = one_hot(gt_mask, height, width)
    gt_mask = make_mask(gt_mask, height, width)

    return gt_onehot, gt_mask, sampling_gt

def print_results(OA_record, kappa_record, alltime_record, dataset_name):
    OA = np.array(OA_record)
    kappa = np.array(kappa_record)
    alltime = np.array(alltime_record)

    print(f'OA={np.mean(OA):.4f} +- {np.std(OA):.4f}')
    print(f'kappa={np.mean(kappa):.4f} +- {np.std(kappa):.4f}')
    print(f'alltime={np.mean(alltime):.4f} +- {np.std(alltime):.4f}')

    with open(f'results/{dataset_name}_results.txt', 'a+') as f:
        str_results = f'\nOA={np.mean(OA):.4f} +- {np.std(OA):.4f}' \
                      f'\nkappa={np.mean(kappa):.4f} +- {np.std(kappa):.4f}' \
                      f'\nalltime={np.mean(alltime):.4f} +- {np.std(alltime):.4f}'
        f.write(str_results)
def generate_png(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)
    numlabel = np.where(numlabel > 1, 0, 1)
    plt.imshow(numlabel, cmap='gray')
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)