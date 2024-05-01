import argparse
import torch
import numpy as np
import utils1
from dataset import Dateset_mat
from tqdm import trange
from model import Encoder_Special, UD_constraint
from utils1 import data_loder
from lightly.loss import NTXentLoss
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'flickr', type=str)
parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--fea_dim", type=int, default=128)
parser.add_argument("--temperature", type=int, default=0.05)
parser.add_argument("--batch_size", type=int, default=256)

config = parser.parse_args()
config.max_ACC = 0
Dataset = Dateset_mat(config.dataset_root)
dataset = Dataset.getdata()
label1 = np.array(dataset[2]) - 1
all_label = np.squeeze(label1)
cluster_num = max(all_label) + 1

max_ACC = 0
NTX_loss = NTXentLoss()

def run_S():
    all_img = (torch.tensor(dataset[0], dtype=torch.float32)).to(device)
    all_txt = torch.tensor(dataset[1], dtype=torch.float32).to(device)
    img1, txt1, label = dataset[0], dataset[1], dataset[2]
    all_label = np.squeeze(label)

    print("clustering number: ", cluster_num)
    data = data_loder(config.batch_size)
    data.get_data(img1, txt1, label)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model = Encoder_Special(all_img.size(1), all_txt.size(1), config.fea_dim, cluster_num, config.batch_size).to(device)
    optimiser_S = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in trange(config.num_epochs):
        model.train()
        model.zero_grad()
        for img, txt, label in data:
            img, txt = img.to(device), txt.to(device)
            img_fea, txt_fea, img_cluster, txt_cluster = model(img, txt, epoch)

            loss2 = NTX_loss(img_fea, txt_fea)
            loss3 = NTX_loss(img_cluster, txt_cluster)

            UDC_img = UD_constraint(img_cluster).to(device)
            UDC_txt = UD_constraint(txt_cluster).to(device)
            loss1 = criterion(img_cluster, UDC_img) + criterion(txt_cluster, UDC_txt)

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimiser_S.step()

        if epoch % 10 == 0:
            acc1, nmi1, acc2, nmi2 = get_S_ACC(model, all_img, all_txt, all_label, epoch)
            print("S: acc1 %.4f nmi1 %.4f acc2 %.4f nmi2 %.4f "% (acc1, nmi1, acc2, nmi2))


def get_S_ACC(model, data1, data2, label, epoch):
    model.eval()
    _, _,  x_out1, x_out2 = model(data1, data2, epoch)
    pre_label = np.array(x_out1.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc1 = utils1.metrics.acc(pre_label, label)
    nmi1 = utils1.metrics.nmi(pre_label, label)

    pre_label = np.array(x_out2.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc2 = utils1.metrics.acc(pre_label, label)
    nmi2 = utils1.metrics.nmi(pre_label, label)
    return acc1, nmi1, acc2, nmi2


if __name__ == '__main__':
    run_S()
