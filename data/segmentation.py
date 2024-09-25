
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def LDA(data, gt, height, width, bands):
    data_2d = np.reshape(data, [height * width, bands])
    gt = np.reshape(gt, [-1])
    index = np.where(gt != 0)[0]
    data_samples = data_2d[index]
    gt_samples = gt[index]

    lda = LinearDiscriminantAnalysis()
    lda.fit(data_samples, gt_samples - 1)
    LDA_result = lda.transform(data_2d)
    print(f"LDA result shape: {LDA_result.shape}")
    num_lda_components = LDA_result.shape[1]  

    if num_lda_components == 1:
        return np.reshape(LDA_result, [height, width])
    else:
        return np.reshape(LDA_result, [height, width, num_lda_components])


def seg_module(data, gt, superpixel_num):
    height, width, bands = data.shape
    LDA_result = LDA(data, gt, height, width, bands)
    if LDA_result.ndim == 1:
        LDA_result = LDA_result[:, np.newaxis] 
    LDA_result_reshaped = np.reshape(LDA_result, (height, width, -1))

    segments = slic(LDA_result_reshaped, n_segments=int(superpixel_num), compactness=0.5)
    superpixel_count = segments.max() + 1
    segments_1d = np.reshape(segments, [-1])

    S = np.zeros([superpixel_count, 1], dtype=np.float32)
    Q = np.zeros([height * width, superpixel_count], dtype=np.float32)   
    LDA_result_flat = LDA_result.reshape(-1)  

    for i in range(superpixel_count):
        idx = np.where(segments_1d == i)[0]
        count = len(idx)
        if count > 0: 
            pixels = LDA_result_flat[idx]
            superpixel = np.sum(pixels) / count  
            S[i] = superpixel
            Q[idx, i] = 1

    A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
    for i in range(height - 2):
        for j in range(width - 2):
            sub = segments[i:i + 2, j:j + 2]
            sub_max = np.max(sub).astype(np.int32)
            sub_min = np.min(sub).astype(np.int32)
            if sub_max != sub_min:
                idx1 = sub_max
                idx2 = sub_min
                if A[idx1, idx2] != 0:
                    continue
                pix1 = S[idx1]
                pix2 = S[idx2]
                diss = np.exp(-np.sum(np.square(pix1 - pix2)) / 10 ** 2)
                A[idx1, idx2] = A[idx2, idx1] = diss

    A = preprocessing.scale(A, axis=1)
    A = A + np.eye(superpixel_count)

    return Q, S, A, segments, superpixel_count
