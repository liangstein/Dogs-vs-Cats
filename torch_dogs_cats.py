import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.optim as optim;
import torch.nn.functional as F;
import os
import random
import pickle;
import numpy as np;
from tqdm import tqdm;
DIR=os.getcwd()
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,16,5,padding=2)
        nn.init.xavier_uniform(self.conv1.weight.data);
        nn.init.constant(self.conv1.bias.data, 0);
        self.conv2=nn.Conv2d(16,16,5,stride=2,padding=2)
        nn.init.xavier_uniform(self.conv2.weight.data);
        nn.init.constant(self.conv2.bias.data, 0);
        self.conv3=nn.Conv2d(16,32,5,padding=2)
        nn.init.xavier_uniform(self.conv3.weight.data);
        nn.init.constant(self.conv3.bias.data, 0);
        self.conv4=nn.Conv2d(32,32,5,stride=2,padding=2)
        nn.init.xavier_uniform(self.conv4.weight.data);
        nn.init.constant(self.conv4.bias.data, 0);
        self.conv5=nn.Conv2d(32,64,5,padding=2)
        nn.init.xavier_uniform(self.conv5.weight.data);
        nn.init.constant(self.conv5.bias.data, 0);
        self.conv6=nn.Conv2d(64,64,5,stride=2,padding=2)
        nn.init.xavier_uniform(self.conv6.weight.data);
        nn.init.constant(self.conv6.bias.data, 0);
        self.conv7 = nn.Conv2d(64, 128, 5, padding=2)
        nn.init.xavier_uniform(self.conv7.weight.data);
        nn.init.constant(self.conv7.bias.data, 0);
        self.conv8= nn.Conv2d(128, 128, 5, stride=2, padding=2)
        nn.init.xavier_uniform(self.conv8.weight.data);
        nn.init.constant(self.conv8.bias.data, 0);
        self.fc1=nn.Linear(25088,1024)
        nn.init.xavier_uniform(self.fc1.weight.data);
        nn.init.constant(self.fc1.bias.data, 0);
        self.fc2=nn.Linear(1024,2)
        nn.init.xavier_uniform(self.fc2.weight.data);
        nn.init.constant(self.fc2.bias.data, 0);
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5=nn.BatchNorm1d(1024)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.bn1(self.conv2(x)))
        #print(x.size())
        x=F.relu(self.conv3(x))
        x=F.relu(self.bn2(self.conv4(x)))
        #print(x.size())
        x=F.relu(self.conv5(x))
        x=F.relu(self.bn3(self.conv6(x)))
        #print(x.size())
        x=F.relu(self.conv7(x))
        x=F.relu(self.bn4(self.conv8(x)))
        #print(x.size())
        x=x.view(-1,self.flattern_dim(x))
        #print(x.size())
        x=F.relu(self.bn5(self.fc1(x)))
        x=self.fc2(x)
        return x
    def flattern_dim(self,x):
        all_shape=x.size()[1:]
        count=1;
        for j in all_shape:
            count*=j
        return count


net=Net()
net.cuda()
net.load_state_dict(torch.load("net"))
params=list(net.parameters());
#test_input=Variable(torch.FloatTensor(10,3,224,224))
#test_output=net(test_input)
batch_size=200
initial_lr=0.0001;
opt=optim.Adam(net.parameters(),lr=initial_lr)

x_train=np.load("dogs_cats_image.npy")
y_train=np.load("dogs_cats_label.npy")
#y_train=np.array([0]*12500+[1]*12500)
def batch_right_rate(output_label,y_variable):
    predict_max_pos=torch.max(output_label,1)[1]
    right_vector=torch.eq(predict_max_pos,y_variable)
    count=torch.sum(right_vector)
    count=float(count.cpu().data.numpy())
    return count/len(output_label);


criterion=nn.CrossEntropyLoss()
last_epoch_loss=0
for epoch in range(1000):
    epoch_loss_list=[]
    epoch_right_list=[]
    all_labels=np.arange(0,len(x_train));
    np.random.shuffle(all_labels)
    batched_labels=np.array_split(all_labels,int(len(x_train)/batch_size))
    for label_of_label in tqdm(range(len(batched_labels))):
        batched_label=batched_labels[label_of_label]
        input_image_matrix=np.zeros((batch_size,224,224,3),dtype=np.float32)
        input_label=np.zeros((batch_size),dtype=np.float32)
        for i,ele in enumerate(batched_label):
            input_image_matrix[i]=(x_train[ele]-127.5)/127.5
            input_label[i]=y_train[ele]
        x_variable=Variable(torch.from_numpy(input_image_matrix).permute(0,3,1,2)).type(torch.FloatTensor).cuda()
        y_variable=Variable(torch.from_numpy(input_label)).type(torch.LongTensor).cuda()
        output_label=net(x_variable)
        epoch_right_list.append(batch_right_rate(output_label,y_variable))
        opt.zero_grad()
        loss=criterion(output_label,y_variable)
        loss.backward()
        opt.step()
        loss_here=loss.cpu()
        epoch_loss_list.append(float(loss_here.data.numpy()))
    epoch_loss=np.mean(epoch_loss_list)
    right_rate=np.mean(epoch_right_list)
    with open("epoch_loss","a") as f:
        f.write("epoch: {}, loss: {}, right rate: {} \n".format(str(epoch),str(epoch_loss),str(right_rate)))
    torch.save(net.state_dict(),"net")



# prediction on test set
net.eval()
net.load_state_dict(torch.load("net",map_location=lambda storage,loc:storage))
test_image=np.load("cats.npy")
test_labels=[]
batch_size=1
all_labels=np.arange(0,len(test_image))
batched_labels=np.array_split(all_labels,int(len(test_image)/batch_size))
for label in tqdm(range(len(batched_labels))):
    batch_label=batched_labels[label]
    batch_image=np.zeros((batch_size,224,224,3),dtype=np.float32)
    for i,ele in enumerate(batch_label):
        batch_image[i]=(test_image[ele]-127.5)/127.5
    x_variable=Variable(torch.from_numpy(batch_image).permute(0,3,1,2)).type(torch.FloatTensor)
    output_label=F.softmax(net(x_variable))
    label_vector=torch.max(output_label,1)[1]
    label_vector=list(label_vector.data.numpy())
    test_labels+=label_vector

np.save("test_labels",test_labels)