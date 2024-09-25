import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor, device):
        super(GCNLayer, self).__init__()
        self.A = A
        self.device = device
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(self.device)  # Use self.device here
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec) + self.I
        if model != 'normal':
            A = torch.clamp(A, 0.1)
        A = F.softmax(A, dim=1)
        output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))
        return output, A


class SSConv(nn.Module):
    """
    Spectral-Spatial Convolution
    """
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class ChangeDetectionHMGCF(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q1: torch.Tensor, Q2: torch.Tensor, A1: torch.Tensor, A2: torch.Tensor, device, model='normal'):
        super(ChangeDetectionCEGCN, self).__init__()

        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.Q1 = Q1
        self.Q2 = Q2
        self.A1 = A1
        self.A2 = A2
        self.model = model
        self.device = device 

        epsilon = 1e-10
        self.norm_col_Q1 = Q1 / (torch.sum(Q1, 0, keepdim=True) + epsilon)
        self.norm_col_Q2 = Q2 / (torch.sum(Q2, 0, keepdim=True) + epsilon)

        layers_count = 3

        # Spectra Transformation Sub-Network for both time points
        self.CNN_denoise1 = self._build_cnn_denoise_layers(layers_count, self.channel)
        self.CNN_denoise2 = self._build_cnn_denoise_layers(layers_count, self.channel)

        # Pixel-level Convolutional Sub-Network for both time points
        self.CNN_Branch1 = self._build_cnn_branch_layers(layers_count)
        self.CNN_Branch2 = self._build_cnn_branch_layers(layers_count)

        # Superpixel-level Graph Sub-Network for both time points
        self.GCN_Branch1 = self._build_gcn_branch_layers(layers_count, self.A1)
        self.GCN_Branch2 = self._build_gcn_branch_layers(layers_count, self.A2)

        # Softmax layer for change detection output
        self.Softmax_linear = nn.Sequential(nn.Linear(128 * 2, self.class_count))

    def _build_cnn_denoise_layers(self, layers_count, channel):
        layers = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                layers.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(channel))
                layers.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(channel, 128, kernel_size=(1, 1)))
                layers.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                layers.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128))
                layers.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                layers.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        return layers

    def _build_cnn_branch_layers(self, layers_count):
        layers = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                layers.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=5))
            else:
                layers.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
        return layers

    def _build_gcn_branch_layers(self, layers_count, A):
        layers = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                layers.add_module('GCN_Branch' + str(i), GCNLayer(128, 128, A, self.device))  # Pass device here
            else:
                layers.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, A, self.device))   # Pass device here
        return layers

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # Processing for time point 1
        noise_x1 = self.CNN_denoise1(torch.unsqueeze(x1.permute([2, 0, 1]), 0))
        noise_x1 = torch.squeeze(noise_x1, 0).permute([1, 2, 0])
        clean_x1 = noise_x1
        clean_x1_flatten = clean_x1.reshape([self.height * self.width, -1])
        superpixels_flatten_x1 = torch.mm(self.norm_col_Q1.t(), clean_x1_flatten)
        CNN_result_x1, GCN_result_x1 = self._process_branches(clean_x1, superpixels_flatten_x1, self.CNN_Branch1, self.GCN_Branch1, self.Q1)

        # Processing for time point 2
        noise_x2 = self.CNN_denoise2(torch.unsqueeze(x2.permute([2, 0, 1]), 0))
        noise_x2 = torch.squeeze(noise_x2, 0).permute([1, 2, 0])
        clean_x2 = noise_x2
        clean_x2_flatten = clean_x2.reshape([self.height * self.width, -1])
        superpixels_flatten_x2 = torch.mm(self.norm_col_Q2.t(), clean_x2_flatten)
        CNN_result_x2, GCN_result_x2 = self._process_branches(clean_x2, superpixels_flatten_x2, self.CNN_Branch2, self.GCN_Branch2, self.Q2)

        # Feature fusion and classification
        Y = torch.cat([GCN_result_x1, GCN_result_x2, CNN_result_x1, CNN_result_x2], dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)

        return Y

    def _process_branches(self, hx, superpixels_flatten, CNN_Branch, GCN_Branch, Q):
        # CNN branch processing
        CNN_result = CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([hx.shape[0] * hx.shape[1], -1])

        # GCN branch processing
        H = superpixels_flatten
        for i in range(len(GCN_Branch)):
            H, _ = GCN_Branch[i](H)
        
        GCN_result = torch.matmul(Q, H)
        return CNN_result, GCN_result

