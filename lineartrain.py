import torch
import torch.nn as nn
import numpy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from nor import normalize
import scipy.io as io
import numpy as np
from torch.autograd import Variable
#数据的读入
data = numpy.loadtxt('zhengqi_train.txt')
input_data=data[0:2600,0:38]
#input_data = normalize(input_data)
input_data=torch.tensor(input_data,dtype=torch.float)
label=data[0:2600,38]
label=torch.tensor(label,dtype=torch.float)
# 将标签与数据一一对应
train_ids = TensorDataset(input_data, label)

class LinearRegression(nn.Module):
    def __init__(self,nfeature):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(nfeature,1)  # 输入和输出的维度

    def forward(self, x):
        out = self.linear(x)
        return out

net = LinearRegression(38)
print(net)
riterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)  #定义优化函数
print(optimizer)
num_epochs=1000
for epoch in range(1,num_epochs+1):
    train_loader = DataLoader(dataset=train_ids, batch_size=64, shuffle=True)
    for X, y in train_loader:
        output = net.forward(X)
        l = riterion(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d,loss:%f' % (epoch, l.item()))
#测试数据的读入
test_data=data[2600:2888,0:38]
#test_data=normalize(test_data)
test_data=torch.tensor(test_data,dtype=torch.float)
test_label=data[2600:2888,38]
test_label=torch.tensor(test_label,dtype=torch.float)
output=net.forward(test_data)
l=riterion(output,test_label.view(-1,1))
print('loss:%f'%(l.item()))

#天池测试
data1 = numpy.loadtxt('zhengqi_test.txt')
data1 = normalize(data1)
data1=torch.tensor(data1,dtype=torch.float)
output = net.forward(data1)

result=output.data.numpy()
np.savetxt('result.txt',result)
io.savemat('result.mat',{'result':result})






