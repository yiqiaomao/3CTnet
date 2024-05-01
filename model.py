import torch
import torch.nn as nn
import numpy as np
from transformer import TransformerBlock

class Encoder_Special(nn.Module):
    def __init__(self, in_img, in_txt, fea_dim, cluster_num,batch_size):
        super(Encoder_Special, self).__init__()
        _initialize_weights(self)
        self.in_img = in_img
        self.in_txt = in_txt
        self.fea_dim = fea_dim
        self.cluster_num = cluster_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc1_img = nn.Sequential(
            nn.Linear(self.in_img, int(self.in_img/2)),
            nn.BatchNorm1d(int(self.in_img/2)),
            nn.ReLU(),
            nn.Linear(int(self.in_img / 2), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc1_txt = nn.Sequential(
            nn.Linear(self.in_txt, int(self.in_txt / 2)),
            nn.BatchNorm1d(int(self.in_txt / 2)),
            nn.ReLU(),
            nn.Linear(int(self.in_txt / 2), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.fc2_img = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.fea_dim),
            nn.BatchNorm1d(self.fea_dim),
            nn.ReLU(),
        )

        self.fc2_txt = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.fea_dim),
            nn.BatchNorm1d(self.fea_dim),
            nn.ReLU(),
        )

        self.cluster_img = nn.Sequential(
            nn.Linear(self.fea_dim, self.cluster_num),
            nn.BatchNorm1d(self.cluster_num),
            nn.ReLU(),
        )
        self.cluster_txt = nn.Sequential(
            nn.Linear(self.fea_dim, self.cluster_num),
            nn.BatchNorm1d(self.cluster_num),
            nn.ReLU(),
        )

        self.Transformer = TransformerBlock(model_dim=512, num_heads=8)


    def forward(self, img, txt, epoch):
        a_img = self.fc1_img(img)
        a_txt = self.fc1_txt(txt)

        fea_img = a_img + self.Transformer(a_img, a_txt)
        fea_txt = a_txt + self.Transformer(a_txt, a_img)

        fea_img = self.fc2_img(fea_img)
        fea_txt = self.fc2_txt(fea_txt)

        cluster_img = self.cluster_img(fea_img)
        cluster_txt = self.cluster_txt(fea_txt)

        cluster_img = torch.softmax(cluster_img, dim=1)
        cluster_txt = torch.softmax(cluster_txt, dim=1)

        return fea_img, fea_txt, cluster_img, cluster_txt


def _initialize_weights(self):
    print("initialize")
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            assert (m.track_running_stats == self.batchnorm_track)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 100:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    argmaxes = np.nanargmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL

