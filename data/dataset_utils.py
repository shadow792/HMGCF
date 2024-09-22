import scipy.io as sio
import os
import numpy as np

def load_dataset():
    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets/Santa_Barbara_Dataset")
    data_b_path = os.path.join(dataset_dir, 'barbara_2013.mat')
    data_a_path = os.path.join(dataset_dir, 'barbara_2014.mat')
    gt_path = os.path.join(dataset_dir, 'barbara_gtChanges.mat')

    data_b = sio.loadmat(data_b_path)
    data_a = sio.loadmat(data_a_path)
    gt_mat = sio.loadmat(gt_path)

    data_before = data_b['HypeRvieW']
    data_after = data_a['HypeRvieW']
    gt = gt_mat['HypeRvieW']
    dataset_name = "SantaBarbara"

    height, width, bands = data_before.shape
    if height < 700:
        gt = 2 - gt  # Adjust for different dataset resolutions

    print(f"Dataset: {dataset_name}, Shape: {height}x{width}, Bands: {bands}")
    data_concat = np.concatenate((data_before, data_after), axis=-1)

    # Preprocess the dataset
    data_before = data_before.reshape(height * width, bands)
    data_after = data_after.reshape(height * width, bands)
    data_concat = data_concat.reshape(height * width, 2 * bands)
    
    # Standardize the dataset
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_before = scaler.fit_transform(data_before)
    data_after = scaler.fit_transform(data_after)
    data_concat = scaler.fit_transform(data_concat)
    
    data_before = data_before.reshape(height, width, bands)
    data_after = data_after.reshape(height, width, bands)
    data_concat = data_concat.reshape(height, width, 2 * bands)

    return data_before, data_after, data_concat, gt, dataset_name
