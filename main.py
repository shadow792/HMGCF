import os

import torch
import time
from data.dataset_utils import load_dataset
from data.segmentation import seg_module
from utils.helpers import sampling, get_mask_onehot, print_results
from models.change_detection_model import ChangeDetectionHMGCF
from training_testing.training_utils import train
from training_testing.test import test
import numpy as np

from utils.helpers import generate_png

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scale = 450
    ITER, lr, num_epochs, sampling_rate = 5, 1e-3, 300, 0.2

    # Load the dataset
    data_before, data_after, data_concat, gt, dataset_name = load_dataset()

    # Calculate superpixel number
    superpixel_num = data_before.shape[0] * data_before.shape[1] / scale
    height, width, bands = data_before.shape

    # Records to store results
    OA_record = []
    kappa_record = []
    alltime_record = []

    # Iterations for training and testing
    for index_iter in range(ITER):
        datapre_time_before = time.time()

        # Sampling and mask creation
        train_index, val_index, test_index = sampling("random", sampling_rate, gt)
        train_onehot, train_mask, train_gt = get_mask_onehot(gt, train_index)
        val_onehot, val_mask, val_gt = get_mask_onehot(gt, val_index)
        test_onehot, test_mask, test_gt = get_mask_onehot(gt, test_index)
        datapre_time_after = time.time()

        # Superpixel segmentation
        SS_time_before = time.time()
        Q1, S1, A1, seg1, supernum1 = seg_module(data_before, train_gt, 0.2 * superpixel_num)
        Q2, S2, A2, seg2, supernum2 = seg_module(data_after, train_gt, 0.2 * superpixel_num)
        SS_time_after = time.time()

        # Prepare tensors for training
        Q1, Q2 = torch.from_numpy(Q1).to(device), torch.from_numpy(Q2).to(device)
        A1, A2 = torch.from_numpy(A1).to(device), torch.from_numpy(A2).to(device)
        train_mask, val_mask, test_mask = [torch.from_numpy(m.astype(np.float32)).to(device) for m in [train_mask, val_mask, test_mask]]
        train_onehot, val_onehot, test_onehot = [torch.from_numpy(o.astype(np.float32)).to(device) for o in [train_onehot, val_onehot, test_onehot]]
        train_gt, val_gt, test_gt = [torch.from_numpy(g.astype(np.float32)).to(device) for g in [train_gt, val_gt, test_gt]]

        net_input_before, net_input_after = [torch.from_numpy(np.array(d, np.float32)).to(device) for d in [data_before, data_after]]
        net_input_concat = torch.from_numpy(np.array(data_concat, np.float32)).to(device)

        # Initialize the model
        net = ChangeDetectionCEGCN(height, width, bands, 2, Q1, Q2, A1, A2, device, model='normal')
        net.to(device)

        # Train the model
        train(net, lr, num_epochs, net_input_before, net_input_after, net_input_concat, train_gt, train_onehot, train_mask, val_gt, val_onehot, val_mask, device)

        # Test the model
        pred, test_OA, test_kappa, test_f1, TP, FP, TN, FN = test(net, net_input_before, net_input_after, test_gt, test_onehot, test_mask)

        # Record results
        classification_map = torch.argmax(pred, 1).reshape([height, width]).cpu() + 1
        OA_record.append(test_OA)
        kappa_record.append(test_kappa)
        alltime = datapre_time_after - datapre_time_before + SS_time_after - SS_time_before
        alltime_record.append(alltime)

        # Save the classification map as an image
        generate_png(classification_map, f"results/{dataset_name}_OA_{test_OA:.4f}")

    # Print final results
    print_results(OA_record, kappa_record, alltime_record, dataset_name)

if __name__ == "__main__":
    main()
