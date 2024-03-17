import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
def main():
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) #defines how the transform should take place, ToTensors changes the PIL image to torch tensor, normalization of the tensor with a mean and std of 0.5 for each channel, this normalization normalizes the pixel values in the range of [-1,1]
    trainset=torchvision.datasets.CIFAR10(root='C:\\Documents\\Python_Stuff\\Machine_Learning\\Datasets',train=True,download=True,transform=transform) #torchvision loads the CIFAR dataset and applies the relevant transform
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2) #loads a dataset to allow for iteration over it, number of subprocesses to use for data loading is specified by num_workers
    testset=torchvision.datasets.CIFAR10(root='C:\\Documents\\Python_Stuff\\Machine_Learning\\Datasets',train=False,download=False,transform=transform)
    testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
    #CNN architecture
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
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
    net=Net()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9) #momentum is used to accelerate the gradient descent
    for epoch in range(10): 
        current_loss=.0
        for i,data in enumerate(trainloader,0):
            inputs, labels=data
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            current_loss+=loss.item()
            if i%2000==1999:
                print(f'[{epoch+1}, {i+1}] loss: {current_loss/2000:.3f}')
                current_loss=.0
    torch.save(net.state_dict(),'Trained_Model_Params/cifar_trained_dataset_params.pth')
if __name__=='__main__':
    multiprocessing.freeze_support()
    main()