import torch
import model
import utils
import torch.nn as nn



data_path = "./data/train_data.csv"
device = torch.device("cuda:1")
batch_size = 64
Epoch = 5

# 读取数据，并划分成训练集和测试集
batch_train_inputs, batch_train_targets,test_features,test_targets,batch_count = utils.load_train_data(data_path,batch_size)

bertclassfication = model.BertClassfication().to(device) #实例化
lossfuction = nn.CrossEntropyLoss().to(device) #定义损失函数，交叉熵损失函数
optimizer = torch.optim.Adam(bertclassfication.parameters(),lr=2e-5)

for epoch in range(Epoch):
    los = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        targets = torch.tensor(batch_train_targets[i]).to(device)
        optimizer.zero_grad()#1.梯度置零
        outputs= bertclassfication(inputs,device)#2.模型获得结果
        loss = lossfuction(outputs,targets)#3.计算损失
        loss.backward()#4.反向传播
        optimizer.step()# 5.修改参数，w，b

        los += loss.item() #item()返回loss的值

        if i%5==0:
            print("Epoch:%d,Batch:%d,Loss %.4f" % ((epoch),(i),los/5))
            los = 5

torch.save(bertclassfication.state_dict(),"./model_param/model.pth")

print("----------------------测试准确率-----------------------")
hit = 0 #用来计数，看看预测对了多少个
total = len(test_features) # 看看一共多少例子
for i in range(total):
    outputs = bertclassfication([test_features[i]],device)
    _,predict = torch.max(outputs,1)
    if predict==test_targets[i]:# 预测对
        hit+=1
print('准确率为%.4f'%(hit/len(test_features)))