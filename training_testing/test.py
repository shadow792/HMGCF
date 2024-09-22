# test.py
import torch
from sklearn import metrics
import numpy as np

def cal_kappa(network_output, test_gt, test_onehot, zeros, height, width):
    with torch.no_grad():
        test_gt_cpu = test_gt.cpu().detach().numpy().reshape(height, width)
        available_label_idx = (test_gt != 0).float()
        available_label_count = available_label_idx.sum()
        correct_prediction = torch.where(
            torch.argmax(network_output, 1) == torch.argmax(test_onehot, 1), available_label_idx,
            zeros).sum()
        OA = (correct_prediction.cpu() / available_label_count).cpu().numpy()

        output_data = network_output.cpu().numpy().reshape(height * width, 2)
        idx = np.argmax(output_data, axis=-1).reshape(height, width)

        test_pre_label_list = []
        test_real_label_list = []
        for ii in range(height):
            for jj in range(width):
                if test_gt_cpu[ii][jj] != 0:
                    test_pre_label_list.append(idx[ii][jj] + 1)
                    test_real_label_list.append(test_gt_cpu[ii][jj])
        test_pre_label_list = np.array(test_pre_label_list)
        test_real_label_list = np.array(test_real_label_list)

        kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))
        f1 = metrics.f1_score(test_real_label_list, test_pre_label_list, average='weighted')
        confusion = metrics.confusion_matrix(test_real_label_list, test_pre_label_list)
        TN, FP, FN, TP = confusion.ravel()

        print("Test OA=", OA, 'Kappa=', kappa, 'F1 Score=', f1, 'Confusion Matrix=', confusion)
        return OA, kappa, f1, TP, FP, TN, FN

def test(net, net_input_before, net_input_after, test_gt, test_onehot, test_mask):
    torch.cuda.empty_cache()
    height, width = test_onehot.size()[:2]
    zeros = torch.zeros([height * width]).to(device).float()

    test_onehot = test_onehot.reshape([-1, 2])
    test_gt = test_gt.reshape([-1])

    with torch.no_grad():
        model_path = os.path.join("model", "best_model.pt")
        net.load_state_dict(torch.load(model_path))
        net.eval()
        output = net(net_input_before, net_input_after)
        test_OA, test_kappa, test_f1, TP, FP, TN, FN = cal_kappa(output, test_gt, test_onehot, zeros, height, width)
        print("Test OA={}\t Test Kappa={}\t Test F1={}".format(test_OA, test_kappa, test_f1))

    return output, test_OA, test_kappa, test_f1, TP, FP, TN, FN
