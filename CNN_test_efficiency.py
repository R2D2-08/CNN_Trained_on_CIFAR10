import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_dataset=datasets.CIFAR10(root='PATH TO STORE THE DOWNLOADED DATASET',train=False,download=True,transform=transform)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5) # 3 inpt-chnls, 6 outpt-chnls (6 features maps), 5*5 is the size of each feature map
        self.pool=nn.MaxPool2d(2,2) # the frist parameter applies the relevant descaling (pooling window does it), stride 
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120) #fully connected layer
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x))) #pools after relu activation on a convolutional input sequence
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5) #reshaping tensor into 2D tensor with batch size and number of channels*spatial dimensions as the dimensions
        x=F.relu(self.fc1(x)) #F.relu applies a relu activation
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
model=Net()
model.load_state_dict(torch.load('cifar_trained_dataset_params.pth'))
model.eval()
criterion=torch.nn.CrossEntropyLoss()
total_correct=0
total_samples=0
with torch.no_grad():
    for inputs,labels in test_loader:
        outputs=model(inputs)
        _,predicted=torch.max(outputs, 1)
        total_correct+=(predicted==labels).sum().item()
        total_samples+=labels.size(0)
accuracy=total_correct/total_samples
print(f'Accuracy : {accuracy:.2%}')
