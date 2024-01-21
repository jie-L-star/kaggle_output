import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split  # 分离训练集和测试集
import mat73

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
    def __init__(self, in_dim):
        super(net, self).__init__()
        hidden_dim = [512, 1024]
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0])
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1])
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim[1], 4)
        )

    def forward(self, x):  # x为整个网络的输入，前向网络调用上述定义的网络层
        d1 = F.relu(self.layer1(x))
        d2 = F.relu(self.layer2(d1))
        d3 = self.layer3(d2)
        return d3          # d3 为整个网络的输出

class net1(nn.Module):
    def __init__(self):
        super().__init__()
        # layers = [64, 1024, 1024, 512, 512, 512, 512, 1]  # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        layers = [64, 512, 128, 512, 256, 2, 2, 2]
        self.layer1 = nn.Linear(layers[0],
                                layers[1])  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.layer3 = nn.Linear(layers[2], layers[3])
        self.layer4 = nn.Linear(layers[3], layers[4])
        self.layer5 = nn.Linear(layers[4], layers[5])
        self.layer6 = nn.Linear(layers[5], layers[6])
        self.layer7 = nn.Linear(layers[6], layers[7])
        # self.layer8 = nn.Linear(layers[7], layers[8])
        # self.layer6 = nn.Linear(layers[5], layers[6])
        self.relu = nn.ReLU()  # 非线性的激活函数。如果只有线性层，那么相当于输出只是输入做了了线性变换的结果，对于线性回归没有问题。但是非线性回归我们需要加入激活函数使输出的结果具有非线性的特征
        self.S = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)


    def forward(self, d):  # d就是整个网络的输入
        d1 = self.layer1(d)
        d1 = self.S(d1)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。

        d2 = self.layer2(d1)  # 但是在网络的最后一层可以不用激活函数，因为有些激活函数会使得输出结果限定在一定的值域里。
        d2 = self.S(d2)
        # d2 = self.dropout(d2)
        d3 = self.layer3(d2)
        d3 = self.S(d3)
        # d3 = self.dropout(d3)
        d41 = self.layer4(d3)
        d4 = self.S(d41)
        # d4 = self.dropout(d4)
        # d4 = self.relu(self.layer4(d3))
        d51 = self.layer5(d4)
        d5 = self.S(d51)
        d61 = self.layer6(d5)
        d6 = (self.S(d61))
        d7 = self.layer7(d6)
        return d51

##
# 输入是sin

class net2(nn.Module):
    def __init__(self):
        super().__init__()
        # layers = [64, 1024, 1024, 512, 512, 512, 512, 1]  # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        layers = [64, 512, 128, 512, 256, 2, 2, 2]
        self.layer1 = nn.Linear(layers[0],
                                layers[1])  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.layer3 = nn.Linear(layers[2], layers[3])
        self.layer4 = nn.Linear(layers[3], layers[4])
        self.layer5 = nn.Linear(layers[4], layers[5])
        self.layer6 = nn.Linear(layers[5], layers[6])
        self.layer7 = nn.Linear(layers[6], layers[7])
        # self.layer8 = nn.Linear(layers[7], layers[8])
        # self.layer6 = nn.Linear(layers[5], layers[6])
        self.relu = nn.ReLU()  # 非线性的激活函数。如果只有线性层，那么相当于输出只是输入做了了线性变换的结果，对于线性回归没有问题。但是非线性回归我们需要加入激活函数使输出的结果具有非线性的特征
        self.S = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)


    def forward(self, d_all):  # d就是整个网络的输入
        d = d_all[:, 0:64]
        d1 = self.layer1(d)
        d1 = self.S(d1)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。

        d2 = self.layer2(d1)  # 但是在网络的最后一层可以不用激活函数，因为有些激活函数会使得输出结果限定在一定的值域里。
        d2 = self.S(d2)
        # d2 = self.dropout(d2)
        d3 = self.layer3(d2)
        d3 = self.S(d3)
        # d3 = self.dropout(d3)
        d41 = self.layer4(d3)
        d4 = self.S(d41)
        # d4 = self.dropout(d4)
        # d4 = self.relu(self.layer4(d3))
        d51 = self.layer5(d4)
        d5 = self.S(d51)
        d61 = self.layer6(d5)
        d6 = (self.S(d61))
        d7 = self.layer7(d6)
        return d51

class net3(nn.Module):
    def __init__(self):
        super().__init__()
        # layers = [64, 1024, 1024, 512, 512, 512, 512, 1]  # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        layers = [64, 1024, 512, 512, 256, 2, 2, 2]
        self.layer1 = nn.Linear(layers[0],
                                layers[1])  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.layer3 = nn.Linear(layers[2], layers[3])
        self.layer4 = nn.Linear(layers[3], layers[4])
        self.layer5 = nn.Linear(layers[4], layers[5])
        self.layer6 = nn.Linear(layers[5], layers[6])
        self.layer7 = nn.Linear(layers[6], layers[7])
        # self.layer8 = nn.Linear(layers[7], layers[8])
        # self.layer6 = nn.Linear(layers[5], layers[6])
        self.relu = nn.ReLU()  # 非线性的激活函数。如果只有线性层，那么相当于输出只是输入做了了线性变换的结果，对于线性回归没有问题。但是非线性回归我们需要加入激活函数使输出的结果具有非线性的特征
        self.S = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, d_all):  # d就是整个网络的输入

        d = d_all[:, 0:64]
        d1 = self.layer1(d)
        d1 = self.S(d1)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。

        d2 = self.layer2(d1)  # 但是在网络的最后一层可以不用激活函数，因为有些激活函数会使得输出结果限定在一定的值域里。
        d2 = self.S(d2)
        # d2 = self.dropout(d2)
        d3 = self.layer3(d2)
        d3 = self.S(d3)
        # d3 = self.dropout(d3)
        d41 = self.layer4(d3)
        d4 = self.S(d41)
        # d4 = self.dropout(d4)
        # d4 = self.relu(self.layer4(d3))
        d51 = self.layer5(d4)

        a = d51[:, 0]
        b = torch.asin(d51[:, 1])
        d511 = torch.zeros_like(d51)
        d511[:, 0] = a
        d511[:, 1] = b * 180 / torch.pi
        R = 6371e3
        c = a / R
        # A1 = torch.acos()
        d5 = self.S(d51)
        d61 = self.layer6(d5)
        d6 = (self.S(d61))
        d7 = self.layer7(d6)
        return d511

class net6(nn.Module):
    def __init__(self):
        super().__init__()
        # layers = [64, 1024, 1024, 512, 512, 512, 512, 1]  # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        layers = [64, 1024, 512, 256, 2, 256, 256, 2]
        self.layer1 = nn.Linear(layers[0],
                                layers[1], dtype=torch.float64)  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2], dtype=torch.float64)
        self.layer3 = nn.Linear(layers[2], layers[3], dtype=torch.float64)
        self.layer4 = nn.Linear(layers[3], layers[4], dtype=torch.float64)
        self.layer5 = nn.Linear(layers[4], layers[5], dtype=torch.float64)
        self.layer6 = nn.Linear(layers[5], layers[6], dtype=torch.float64)
        self.layer7 = nn.Linear(layers[6], layers[7], dtype=torch.float64)
        # self.layer8 = nn.Linear(layers[7], layers[8])
        # self.layer6 = nn.Linear(layers[5], layers[6])
        self.relu = nn.ReLU()  # 非线性的激活函数。如果只有线性层，那么相当于输出只是输入做了了线性变换的结果，对于线性回归没有问题。但是非线性回归我们需要加入激活函数使输出的结果具有非线性的特征
        self.S = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)


    def forward(self, d_all):  # d就是整个网络的输入

        d = d_all[:, 0:64]
        dH = d_all[:, 64]
        d1 = self.layer1(d)
        d1 = self.relu(d1)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。
        d2 = self.layer2(d1)  # 但是在网络的最后一层可以不用激活函数，因为有些激活函数会使得输出结果限定在一定的值域里。
        d2 = self.S(d2)
        # d2 = self.dropout(d2)
        d31 = self.layer3(d2)
        d3 = self.S(d31)
        # d3 = self.dropout(d3)
        d41 = self.layer4(d3)
        # d4 = self.S(d41)
        # # d4 = self.dropout(d4)
        # # d4 = self.relu(self.layer4(d3))
        # d5 = self.layer5(d4)
        #
        #
        #
        # # A1 = torch.acos()
        # d5 = self.S(d5)
        # d61 = self.layer6(d5)
        # d6 = (self.S(d61))
        # d7 = self.layer7(d6)


        d51 = d41
        a = d51[:, 0]
        b = d51[:, 1]
        d511 = torch.zeros_like(d51)
        d511[:, 0] = torch.sqrt(torch.abs(a*a - dH*dH))
        # d511[:, 0] = a
        d511[:, 1] = b   # rad

        Ea = 6378137
        Eb = 6356725
        AW = torch.tensor([28.812340]).to(device=device)
        AJ = torch.tensor([112.403532]).to(device=device)
        dx = d511[:, 0] * torch.cos(d511[:, 1])
        dy = d511[:, 0] * torch.sin(d511[:, 1])

        ec = Eb + (Ea - Eb) * (90 - AW) / 90
        ed = ec * torch.cos(AW * torch.pi / 180)

        BJ = dx / ed * 180 / torch.pi + AJ
        BW = dy / ec * 180 / torch.pi + AW


        d512 = torch.zeros_like(d51)
        d512[:, 0] = BJ
        d512[:, 1] = BW
        d512 = d512*1e3
        return d512

class net7(nn.Module):
    def __init__(self):
        super().__init__()
        # layers = [64, 1024, 1024, 512, 512, 512, 512, 1]  # 网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        layers = [64, 1500, 1024, 1024, 1024, 512, 512, 512, 2]
        self.layer1 = nn.Linear(layers[0],
                                layers[1])  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.layer3 = nn.Linear(layers[2], layers[3])
        self.layer4 = nn.Linear(layers[3], layers[4])
        self.layer5 = nn.Linear(layers[4], layers[5])
        self.layer6 = nn.Linear(layers[5], layers[6])
        self.layer7 = nn.Linear(layers[6], layers[7])
        self.layer8 = nn.Linear(layers[7], layers[8])
        # self.layer6 = nn.Linear(layers[5], layers[6])
        self.relu = nn.ReLU()  # 非线性的激活函数。如果只有线性层，那么相当于输出只是输入做了了线性变换的结果，对于线性回归没有问题。但是非线性回归我们需要加入激活函数使输出的结果具有非线性的特征
        self.S = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)


    def forward(self, d_all):  # d就是整个网络的输入

        d = d_all[:, 0:64]
        dH = d_all[:, 64]
        d1 = self.layer1(d)
        d1 = self.S(d1)  # 每一个线性层之后都需要加入一个激活函数使其非线性化。
        d2 = self.layer2(d1)  # 但是在网络的最后一层可以不用激活函数，因为有些激活函数会使得输出结果限定在一定的值域里。
        d2 = self.relu(d2)
        # d2 = self.dropout(d2)
        d31 = self.layer3(d2)
        d3 = self.S(d31)
        # d3 = self.dropout(d3)
        d41 = self.layer4(d3)
        d4 = self.S(d41)
        # # d4 = self.dropout(d4)
        # # d4 = self.relu(self.layer4(d3))
        d5 = self.layer5(d4)
        #
        #
        #
        # # A1 = torch.acos()
        d5 = self.relu(d5)
        d61 = self.layer6(d5)
        d6 = (self.S(d61))
        d71 = self.layer7(d6)
        d7 = self.S(d71)
        d8 = self.layer8(d7)



        d51 = d8
        a = d51[:, 0]
        b = d51[:, 1]
        d511 = torch.zeros_like(d51)
        d511[:, 0] = torch.sqrt(torch.abs(a*a - dH*dH))
        # d511[:, 0] = a
        d511[:, 1] = b*torch.pi/180   # rad

        Ea = 6378137
        Eb = 6356725
        AW = torch.tensor([28.812340])
        AJ = torch.tensor([112.403532])
        dx = d511[:, 0] * torch.cos(d511[:, 1])
        dy = d511[:, 0] * torch.sin(d511[:, 1])

        ec = Eb + (Ea - Eb) * (90 - AW) / 90
        ed = ec * torch.cos(AW * torch.pi / 180)

        BJ = dx / ed * 180 / torch.pi + AJ
        BW = dy / ec * 180 / torch.pi + AW


        d512 = torch.zeros_like(d51)
        d512[:, 0] = BJ
        d512[:, 1] = BW
        d512 = d512*1e5
        return d512

class  LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        layers = [2, 256, 64, 2, 1024, 512, 512, 512, 2]
        self.layer1 = nn.Linear(layers[0],
                                layers[1])  # 用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.layer3 = nn.Linear(layers[2], layers[3])
        self.layer4 = nn.Linear(layers[3], layers[4])
        self.layer5 = nn.Linear(layers[4], layers[5])
        self.layer6 = nn.Linear(layers[5], layers[6])
        self.layer7 = nn.Linear(layers[6], layers[7])
        self.layer8 = nn.Linear(layers[7], layers[8])

        self.S = nn.Sigmoid()


    def forward(self, d_all):
        d = d_all[:, 0:64].unsqueeze(0)
        dH = d_all[:, 64]
        h0 = torch.zeros(self.num_layers, d.size(1), self.hidden_size).to(d.device)
        c0 = torch.zeros(self.num_layers, d.size(1), self.hidden_size).to(d.device)
        out, _ = self.lstm(d, (h0, c0))
        out1 = self.fc1(out[-1, :, :])
        out2 = self.fc2(out[-1, :, :])
        pred1, pred2 = out1, out2
        pred = torch.stack([pred1[:, -1], pred2[:, -1]], dim=1)

        # c1 = self.S(out)
        # c1 = self.layer1(c1)
        # c2 = self.S(c1)
        # c2 = self.layer2(c2)
        # c3 = self.S(c2)
        # c3 = self.layer3(c3)



        d51 = pred
        a = d51[:, 0]
        b = d51[:, 1]
        d511 = torch.zeros_like(d51)
        d511[:, 0] = torch.sqrt(torch.abs(a * a - dH * dH))
        # d511[:, 0] = a
        d511[:, 1] = b * torch.pi / 180  # rad

        Ea = 6378137
        Eb = 6356725
        AW = torch.tensor([28.812340])
        AJ = torch.tensor([112.403532])
        dx = d511[:, 0] * torch.cos(d511[:, 1])
        dy = d511[:, 0] * torch.sin(d511[:, 1])

        ec = Eb + (Ea - Eb) * (90 - AW) / 90
        ed = ec * torch.cos(AW * torch.pi / 180)

        BJ = dx / ed * 180 / torch.pi + AJ
        BW = dy / ec * 180 / torch.pi + AW

        d512 = torch.zeros_like(d51)
        d512[:, 0] = BJ
        d512[:, 1] = BW
        d512 = d512 * 1e5

        return d512

class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))

        pred1, pred2 = self.fc1(output), self.fc2(output)
        pred1, pred2 = pred1[:, -1, :], pred2[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        pred = torch.stack([pred1, pred2], dim=0)
        # print(pred.shape)

        return pred

class LSTM3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        layers1 = [hidden_size, 512, 256, output_size]
        layers2 = [hidden_size, 512, 256, output_size]

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, dtype=torch.float64)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, dtype=torch.float64)
        self.fc1 = nn.Linear(layers1[0], layers1[1], dtype=torch.float64)
        self.fc11 = nn.Linear(layers1[1], layers1[2], dtype=torch.float64)
        self.fc12 = nn.Linear(layers1[2], layers1[3], dtype=torch.float64)
        self.fc2 = nn.Linear(layers2[0], layers2[1], dtype=torch.float64)
        self.fc21 = nn.Linear(layers2[1], layers2[2], dtype=torch.float64)
        self.fc22 = nn.Linear(layers2[2], layers2[3], dtype=torch.float64)

        self.S = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, d_all):

        # d = d_all[:, 0:64].unsqueeze(0)
        # dH = d_all[:, 64]
        d = d_all.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        c0 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        out, _ = self.lstm1(d, (h0, c0))
        h1 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        c1 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        out_2, _ = self.lstm2(d, (h1, c1))


        out1 = self.fc1(out[-1, :, :])
        out1 = self.S(out1)
        out11 = self.fc11(out1)
        out11 = self.S(out11)
        out12 = self.fc12(out11)

        out2 = self.fc2(out_2[-1, :, :])
        out2 = self.relu(out2)
        out21 = self.fc21(out2)
        out21 = self.relu(out21)
        out22 = self.fc22(out21)

        pred1, pred2 = out12, out22
        pred = torch.stack([pred1[:, -1], pred2[:, -1]], dim=1)

        d51 = pred
        # a = d51[:, 0]
        # b = d51[:, 1]
        # d511 = torch.zeros_like(d51)
        # d511[:, 0] = torch.sqrt(torch.abs(a * a - dH * dH))
        # # d511[:, 0] = a
        # d511[:, 1] = b * torch.pi / 180  # rad
        #
        # Ea = 6378137
        # Eb = 6356725
        # AW = torch.tensor([28.812340]).to(device=device)
        # AJ = torch.tensor([112.403532]).to(device=device)
        # dx = d511[:, 0] * torch.cos(d511[:, 1])
        # dy = d511[:, 0] * torch.sin(d511[:, 1])
        #
        # ec = Eb + (Ea - Eb) * (90 - AW) / 90
        # ed = ec * torch.cos(AW * torch.pi / 180)
        #
        # BJ = dx / ed * 180 / torch.pi + AJ
        # BW = dy / ec * 180 / torch.pi + AW
        #
        # d512 = torch.zeros_like(d51)
        # d512[:, 0] = BJ
        # d512[:, 1] = BW
        # d512 = d512*1e4

        return d51



class LSTM4(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        layers1 = [hidden_size, 512, 256, output_size]
        layers2 = [hidden_size, 512, 256, output_size]

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, dtype=torch.float64)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, dtype=torch.float64)
        self.fc1 = nn.Linear(layers1[0], layers1[1], dtype=torch.float64)
        self.fc11 = nn.Linear(layers1[1], layers1[2], dtype=torch.float64)
        self.fc12 = nn.Linear(layers1[2], layers1[3], dtype=torch.float64)
        self.fc2 = nn.Linear(layers2[0], layers2[1], dtype=torch.float64)
        self.fc21 = nn.Linear(layers2[1], layers2[2], dtype=torch.float64)
        self.fc22 = nn.Linear(layers2[2], layers2[3], dtype=torch.float64)

        self.S = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, d_all):

        # d = d_all[:, 0:64].unsqueeze(0)
        # dH = d_all[:, 64]
        d = d_all.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        c0 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        out, _ = self.lstm1(d, (h0, c0))
        h1 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        c1 = torch.zeros(self.num_layers, d.size(1), self.hidden_size, dtype=torch.float64).to(device=device)
        out_2, _ = self.lstm2(d, (h1, c1))


        out1 = self.fc1(out[-1, :, :])
        out1 = self.S(out1)
        out11 = self.fc11(out1)
        out11 = self.S(out11)
        out12 = self.fc12(out11)

        out2 = self.fc2(out_2[-1, :, :])
        out2 = self.relu(out2)
        out21 = self.fc21(out2)
        out21 = self.relu(out21)
        out22 = self.fc22(out21)

        pred1, pred2 = out12, out22
        pred = torch.cat([pred1, pred2], dim=1)

        d51 = pred
        # a = d51[:, 0]
        # b = d51[:, 1]
        # d511 = torch.zeros_like(d51)
        # d511[:, 0] = torch.sqrt(torch.abs(a * a - dH * dH))
        # # d511[:, 0] = a
        # d511[:, 1] = b * torch.pi / 180  # rad
        #
        # Ea = 6378137
        # Eb = 6356725
        # AW = torch.tensor([28.812340]).to(device=device)
        # AJ = torch.tensor([112.403532]).to(device=device)
        # dx = d511[:, 0] * torch.cos(d511[:, 1])
        # dy = d511[:, 0] * torch.sin(d511[:, 1])
        #
        # ec = Eb + (Ea - Eb) * (90 - AW) / 90
        # ed = ec * torch.cos(AW * torch.pi / 180)
        #
        # BJ = dx / ed * 180 / torch.pi + AJ
        # BW = dy / ec * 180 / torch.pi + AW
        #
        # d512 = torch.zeros_like(d51)
        # d512[:, 0] = BJ
        # d512[:, 1] = BW
        # d512 = d512*1e4

        return d51

def ar2JW(x, dH):
    dH = dH.reshape(-1)
    d51 = x
    a = d51[:, 0]
    b = d51[:, 1]
    d511 = torch.zeros_like(d51)
    d511[:, 0] = torch.sqrt(torch.abs(a * a - dH * dH))
    # d511[:, 0] = a
    d511[:, 1] = b * torch.pi / 180  # rad

    Ea = 6378137
    Eb = 6356725
    AW = torch.tensor([28.812340]).to(device=device)
    AJ = torch.tensor([112.403532]).to(device=device)
    dx = d511[:, 0] * torch.cos(d511[:, 1])
    dy = d511[:, 0] * torch.sin(d511[:, 1])

    ec = Eb + (Ea - Eb) * (90 - AW) / 90
    ed = ec * torch.cos(AW * torch.pi / 180)

    BJ = dx / ed * 180 / torch.pi + AJ
    BW = dy / ec * 180 / torch.pi + AW

    d512 = torch.zeros_like(d51)
    d512[:, 0] = BJ
    d512[:, 1] = BW
    d512 = d512

    return d512

class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=0, dtype=torch.float64)
        self.norm1 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2= nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0, dtype=torch.float64)
        self.norm2 = nn.BatchNorm2d(256, dtype=torch.float64)
        self.fla = nn.Flatten()
        self.relu = nn.ReLU()
        self.S = nn.Sigmoid()
        self.fc1 = nn.Linear(4096, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc4 = nn.Linear(256, 2, dtype=torch.float64)
        self.lstm = LSTM3(4096, 512,1024,64)



    def forward(self, dx):
        x = dx[:, 0:4, :, :]
        x1 = self.S(self.norm1(self.conv1(x)))
        x2 = self.relu(self.norm2(self.conv2(x1)))
        x3 = self.fla(x2)
        x4 = self.S(self.fc1(x3))
        x5 = self.S(self.fc2(x4))
        x6 = self.relu(self.fc3(x5))

        x7 = self.fc4(x6)
        # x7 = ar2JW(x6, dx[:, 3, 1, 1])

        return x7

class Mcd(nn.Module):
    def __init__(self):
        super(Mcd,self).__init__()
        self.con1=nn.Conv2d(3, 32, 2, padding=0,dtype=torch.float64)
        self.maxp1=nn.MaxPool2d(2)
        self.con2=nn.Conv2d(32, 32, 2, padding=0,dtype=torch.float64)
        self.maxp2=nn.MaxPool2d(2,)
        self.con3=nn.Conv2d(32, 64, 3, padding=0,dtype=torch.float64)
        self.maxp3=nn.MaxPool2d(2)
        self.fla=nn.Flatten()
        self.lin1=nn.Linear(1024, 64,dtype=torch.float64)
        self.lin2=nn.Linear(64, 2,dtype=torch.float64)
    def forward(self,dx):
        x = dx[:, 0:3, :, :]
        x=self.con1(x)
        x=self.maxp1(x)
        x=self.con2(x)
        x=self.maxp2(x)
        x=self.con3(x)
        x=self.maxp3(x)
        x=self.fla(x)
        x=self.lin1(x)
        x=self.lin2(x)

        return x

class cnn2(nn.Module):
    def __init__(self):
        super(cnn2,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=5, stride=1, padding=2, dtype=torch.float64)
        self.norm1 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2= nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, dtype=torch.float64)
        self.norm2 = nn.BatchNorm2d(256, dtype=torch.float64)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=0, dtype=torch.float64)
        self.norm3 = nn.BatchNorm2d(128, dtype=torch.float64)
        self.fla = nn.Flatten()
        self.relu = nn.ReLU()
        self.S = nn.Sigmoid()
        self.fc1 = nn.Linear(2048, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc4 = nn.Linear(256, 2, dtype=torch.float64)
        # self.lstm = LSTM4(4096, 64,200,128)



    def forward(self, dx):
        x = dx[:, 0:3, :, :]
        x1 = self.S(self.norm1(self.conv1(x)))
        x2 = self.S(self.norm2(self.conv2(x1)))

        # x41 = self.lstm(x3)
        x3 = self.S(self.norm3(self.conv3(x2)))
        x42 = self.fla(x3)
        x4 = self.relu(self.fc1(x42))
        x5 = self.relu(self.fc2(x4))
        x6 = self.relu(self.fc3(x5))

        x7 = self.fc4(x6)
        # x7 = ar2JW(x6, dx[:, 3, 1, 1])

        return x7


class cnn3(nn.Module):
    def __init__(self):
        super(cnn3,  self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=5, stride=1, padding=2, dtype=torch.float64)
        self.norm1 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, dtype=torch.float64)
        self.norm2 = nn.BatchNorm2d(256, dtype=torch.float64)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0, dtype=torch.float64)
        self.norm3 = nn.BatchNorm2d(128, dtype=torch.float64)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0,
                               dtype=torch.float64)
        self.norm4 = nn.BatchNorm2d(128, dtype=torch.float64)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=2,
                               dtype=torch.float64)
        self.norm5 = nn.BatchNorm2d(64, dtype=torch.float64)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0,
                               dtype=torch.float64)

        self.norm6 = nn.BatchNorm2d(64, dtype=torch.float64)

        self.dropout = nn.Dropout(0.5)

        self.fla = nn.Flatten()
        self.relu = nn.ReLU()
        self.S = nn.Sigmoid()
        self.fc1 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc4 = nn.Linear(256, 2, dtype=torch.float64)
        # self.lstm = LSTM4(4096, 64,200,128)



    def forward(self, dx):
        x = dx[:, 0:3, :, :]
        x1 = self.S(self.norm1(self.conv1(x)))
        x2 = self.S(self.norm2(self.conv2(x1)))

        # x41 = self.lstm(x3)
        x3 = self.S(self.norm3(self.conv3(x2)))
        x31 = self.relu(self.norm4(self.conv4(x3)))
        x31 = self.dropout(x31)
        x32 = self.relu(self.norm5(self.conv5(x31)))
        x32 = self.dropout(x32)
        x33 = self.S(self.norm6(self.conv6(x32)))
        x42 = self.fla(x33)
        x4 = self.relu(self.fc1(x42))
        x4 = self.dropout(x4)
        x5 = self.relu(self.fc2(x4))
        x5 = self.dropout(x5)
        x6 = self.relu(self.fc3(x5))

        x7 = self.fc4(x6)
        # x7 = ar2JW(x6, dx[:, 3, 1, 1])

        return x7

class cnn_ANGLE(nn.Module):
    def __init__(self):
        super(cnn_ANGLE,  self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=5, stride=1, padding=2, dtype=torch.float64)
        self.norm1 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dtype=torch.float64)
        self.norm2 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0, dtype=torch.float64)
        self.norm3 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0,
                               dtype=torch.float64)
        self.norm4 = nn.BatchNorm2d(512, dtype=torch.float64)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=2,
                               dtype=torch.float64)
        self.norm5 = nn.BatchNorm2d(256, dtype=torch.float64)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=0,
                               dtype=torch.float64)

        self.norm6 = nn.BatchNorm2d(64, dtype=torch.float64)

        self.dropout = nn.Dropout(0.5)

        self.fla = nn.Flatten()
        self.relu = nn.ReLU()
        self.S = nn.Sigmoid()
        self.fc1 = nn.Linear(1024, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 512, dtype=torch.float64)
        self.fc3 = nn.Linear(512, 256, dtype=torch.float64)
        self.fc4 = nn.Linear(128, 1, dtype=torch.float64)
        # self.lstm = LSTM4(4096, 64,200,128)



    def forward(self, dx):
        x = dx[:, 0:3, :, :]
        x1 = self.relu(self.norm1(self.conv1(x)))
        x2 = self.S(self.norm2(self.conv2(x1)))
        x2 = self.dropout(x2)

        # x41 = self.lstm(x3)
        x3 = self.relu(self.norm3(self.conv3(x2)))
        x3 = self.dropout(x3)
        x31 = self.relu(self.norm4(self.conv4(x3)))
        x31 = self.dropout(x31)
        x32 = self.relu(self.norm5(self.conv5(x31)))
        x32 = self.dropout(x32)
        x33 = self.relu(self.norm6(self.conv6(x32)))
        x42 = self.fla(x33)
        x4 = self.relu(self.fc1(x42))
        x4 = self.dropout(x4)
        x5 = self.relu(self.fc2(x4))
        x5 = self.dropout(x5)
        x6 = self.relu(self.fc3(x5))

        x7 = self.fc4(x6)
        # x7 = ar2JW(x6, dx[:, 3, 1, 1])

        return x7
