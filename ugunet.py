import math
import torch
import os, sys
import argparse
import functools
import cv2 as cv
import skimage.io
import torchvision
import  numpy as np
import torch.nn as nn
import networkx as nx
from torch.nn import init
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch_geometric.nn import GraphUNet
from torch.optim import lr_scheduler, optimizer
from torch.utils.data import DataLoader, sampler
from torch_geometric.utils.convert import from_networkx


class Dataset(object):
    def __init__(self, image_dir, vessel_dir,graph_dir):
        self.images = []
        self.vessels = []
        self.graphs = []
        imgs = os.listdir(image_dir)
        vess = os.listdir(vessel_dir)
        gph = os.listdir(graph_dir)
        for i in range(len(imgs)):
            img_file = os.path.join(image_dir, imgs[i])
            vess_file = os.path.join(vessel_dir, vess[i])
            gph_file = os.path.join(graph_dir, gph[i])
            # print(img_file, mask_file)
            self.images.append(img_file)
            self.vessels.append(vess_file)
            self.graphs.append(gph_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            vessel_path = self.vessels[idx]
            graph_path = self.graphs[idx]
        else:
            image_path = self.images[idx]
            vessel_path = self.vessels[idx]
            graph_path = self.graphs[idx]


        len_y = len_x = 592

        img = skimage.io.imread(image_path)  # 读入血管图
        img = img.astype(float) / 255
        img = img >= 0.5
        temp = np.copy(img)
        img = np.zeros((592, 592), dtype=temp.dtype)
        img[:temp.shape[0], :temp.shape[1]] = temp
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(dim=0)


        ves = skimage.io.imread(vessel_path)  # 读入原图
        ves = ves.astype(float) / 255
        temp = np.copy(ves)
        temp = temp.transpose(2, 0, 1) #[h,w,c]to [c,h,w]
        ves = np.zeros((3,592, 592), dtype=temp.dtype)
        ves[:,:temp.shape[1], :temp.shape[2]] = temp
        ves = ves.astype(np.float32)
        ves = torch.from_numpy(ves)
        print(ves.size())
        #ves = ves.unsqueeze(dim=0)

        #gph = nx.read_gpickle(graph_path)


        sample = {'image': torch.Tensor(img), 'vessel': torch.Tensor(ves), 'graph':graph_path}

        return sample

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class NetG(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        #self.layer5 = self.base_layers[7]
        #self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.gu = GraphUNet(256, 64, 256, depth=2)

    def forward(self, input_i,input_g):
        e1 = self.layer1(input_i) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        f = self.layer4(e3)

        input_graph = nx.read_gpickle(input_g[0])
        g_in = self.map_graph(f,input_graph) #make graph
        x = g_in.x.cuda()
        edge_index =g_in.edge_index.cuda()
        g_f = self.gu(x, edge_index)
        #g_f_sf = F.log_softmax(g_f, dim=1)#output list
        #print(g_f_sf.size())
        g_out = self.map_matix(input_graph,g_f,f)

        #d4 = self.decode4(f,e4)
        #print(d4.size())
        d3 = self.decode3(g_out, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out

    def map_graph(self,T, G):
        node_list = list(G.nodes)
        for i, n in enumerate(node_list):
            x = math.ceil(G.node[n]['pos'][1] / 16)
            y = math.ceil(G.node[n]['pos'][0] / 16)
            tmp = T[:, :, y, x]
            tmp = tmp.squeeze(0)
            tmp = tmp.detach().cpu()
            tmp = tmp.numpy()
            G.add_node(n, x=tmp)

            # print(G.node[n]['x'])
        G = from_networkx(G)
        return G

    def map_matix(self,G, out_list, T):
        for i, n in enumerate(out_list):
            x = math.ceil(G.node[i]['pos'][1] / 16)
            y = math.ceil(G.node[i]['pos'][0] / 16)
            clone_T = T.clone()
            clone_T[:, :, y, x] = out_list[i]
        return clone_T

# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()  # 对继承自父类的属性进行初始化
        # layer1 输入 3 x 592 x 592, 输出 (ndf) x 286 x 286
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 148 x 148
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 74 x 74
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 37 x 37
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 3, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，是真是假
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )
        # layer5 输出一个数(概率)
        self.layer6 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，是真是假
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    # 定义NetD的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--imageSize', type=int, default=592)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--outf', default='./gresult/', help='folder to output images and model checkpoints')
opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = './dataset/retinal/'
vessel_dir = './dataset/vessel/'
graph_dir = './dataset/graph/'
dataloader = Dataset(image_dir, vessel_dir, graph_dir)
train_loader = DataLoader(dataloader, batch_size=1, shuffle=False)

netG = NetG(3).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(1, opt.epoch + 1):
    for i, sample_batched in enumerate(train_loader):  # 每次epoch，遍历所有图片，共800个batch
        images_batch, target_labels, graph_batch =    sample_batched['image'], sample_batched['vessel'], sample_batched['graph']
        #image:血管； target:眼底 graph:图
        # 1,固定生成器G，训练鉴别器D
        real_label = Variable(torch.ones(opt.batchSize)).cuda()
        fake_label = Variable(torch.zeros(opt.batchSize)).cuda()

        netD.zero_grad()

        # 让D尽可能的把真图片判别为1
        real_imgs = Variable(target_labels.to(device))
        real_output = netD(real_imgs)
        d_real_loss = criterion(real_output, real_label)
        real_scores = real_output
        # d_real_loss.backward()  # compute/store gradients, but don't change params

        # 让D尽可能把假图片判别为0
        '''noise = Variable(torch.randn(opt.batchSize, opt.nz, 1, 1)).to(device)
        noise = noise.to(device)'''
        fake_imgs = netG(images_batch.to(device),graph_batch)  # 生成假图
        fake_output = netD(fake_imgs.detach())  # 避免梯度传到G，因为G不用更新, detach分离
        d_fake_loss = criterion(fake_output, fake_label)
        fake_scores = fake_output
        # d_fake_loss.backward()

        d_total_loss = d_fake_loss + d_real_loss
        netG.zero_grad()
        d_total_loss.backward()  # 反向传播，计算梯度
        optimizerD.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        # 2,固定鉴别器D，训练生成器G
        fake_output = netD(fake_imgs)
        g_fake_loss = criterion(fake_output, real_label)
        g_fake_loss.backward()  # 反向传播，计算梯度
        optimizerG.step()  # 梯度信息来更新网络的参数，Only optimizes G's parameters

        print('[%d/%d][%d/%d] real_scores: %.3f fake_scores %.3f'
              % (epoch, opt.epoch, i, len(dataloader), real_scores.data.mean(), fake_scores.data.mean()))
        if i % 100 == 0:
            vutils.save_image(fake_imgs.data,
                              '%s/fake_samples_epoch_%03d_batch_i_%03d.png' % (opt.outf, epoch, i),
                              normalize=True)
    if epoch/100 ==0:
            torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))