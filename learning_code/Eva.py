import mat73
import torch

from MODEL import *     # 从模型文件中加载定义的网络架构

# 训练前准备，检测是否有可用的GPU，有则使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataMat = mat73.loadmat('D:/desktop/deep learning/pytorch_zj/deeplearning1/data/dataJW.mat')
lllabel = mat73.loadmat('D:/desktop/deep learning/pytorch_zj/deeplearning1/data/realar_test.mat')


# signal_train = torch.from_numpy(dataMat['train_signal'])
signal_test = torch.from_numpy(dataMat['test_signal'])
# label_test_all = torch.from_numpy(dataMat['ltest'])
label_test_all = torch.from_numpy(dataMat['ltest'])
# label_test = label_test_all[:, 0:2]
dH_test = label_test_all[:, 2]

label_test = np.zeros_like(label_test_all[:, 0:2])
label_test[:, 0] = torch.from_numpy(lllabel['c'])
label_test[:, 1] = torch.from_numpy(lllabel['dL'])
# label_train = label_train_all[:, 0:2]
# dH_test = label_test_all[:, 2]

Array_num = signal_test.shape[0]
Snap_num = signal_test.shape[1]
# Sample_train = signal_test.shape[2]
Sample_test = signal_test.shape[2]

Snap_len = 2000

H = np.zeros((Array_num, Snap_num), dtype=complex)
X = np.zeros((Sample_test, 4, Array_num, Array_num), dtype=float)
for Sample_index in range(1, Sample_test + 1):
    H = signal_test[:, 0:Snap_len, Sample_index-1]
    R_temp = 1 / Snap_len * torch.mm(H, torch.adjoint(H))

    X[Sample_index - 1, 0, :, :] = torch.real(R_temp)
    X[Sample_index - 1, 1, :, :] = torch.imag(R_temp)
    X[Sample_index - 1, 2, :, :] = torch.angle(R_temp)
    X[Sample_index - 1, 3, :, :] = torch.ones_like(R_temp) * dH_test[Sample_index-1]

# X[:, Array_num*Array_num] = dH_train
# X_train, X_test, Label_train, Label_test = train_test_split(X, label_test, test_size=0.01)
# scales = MinMaxScaler(feature_range=(0, 1))
# X_train_s = scales.fit_transform(X_train)
# X_test_s = scales.fit_transform(X_test)   # 对数据归一化处理
# 将数据转为张量
X_test_nots = torch.from_numpy(X.astype(np.float64)).to(device=device)

Label_test_t = torch.from_numpy(label_test.astype(np.float64)).to(device=device)

# 将训练集转化为张量后，使用TensorDataset将输入和标签整理到一起
test_data_nots = TensorDataset(X_test_nots, Label_test_t)

##
# 选取网络，定义网络超参数
input_dim = 64
hidden_dim = 20
hidden_layer = 10
out_dim = 1


# Model = LSTM3(input_dim, hidden_dim, hidden_layer, out_dim)   # 选取网络，括号中为输入信号的维度
Model = cnn3()
# Model = net6()
Model.to(device)
Loss = nn.MSELoss()

Model.load_state_dict(torch.load("./model/model_iter4.pth", map_location=device))#pytoch 导入模型
# Dnn.eval()# 这里指评价模型，不反传，所以用eval模式
pt_y_test1 = Model(X_test_nots)
Label = Label_test_t.cpu().numpy()
lll = len(Label)
pt_y_test = torch.zeros((lll, 2)).to(device=device)
pt_y_test[:, 0] = pt_y_test1[:, 0]
pt_y_test[:, 1] = pt_y_test1[:, 1]

y_test = pt_y_test.detach().cpu()  # 输出结果torch tensor，需要转化为numpy类型来进行可视
Y = y_test.numpy()
LOss1 = Loss(pt_y_test[:, 0], Label_test_t[:, 0])
LOss2 = Loss(pt_y_test[:, 1], Label_test_t[:, 1])

plt.figure()
T = np.linspace(1, lll, lll)
Diff = Y - Label
plt.plot(T, Diff[:, 0])

plt.figure()
plt.plot(T, Diff[:, 1])
plt.show()

# 网络超参数

# 可视化训练及测试损失值
plt.title('trainloss')
plt.plot(np.arange(len(losses_J)), losses_J)
plt.plot(np.arange(len(losses_W)), losses_W)
plt.plot(np.arange(len(eval_losses_J)), eval_losses_J)
plt.plot(np.arange(len(eval_losses_W)), eval_losses_W)
plt.legend(['Test Loss J', 'Test Loss W'], loc='upper right')
# plt.legend(['Test Loss'], loc='upper right')

plt.show()
