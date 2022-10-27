import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import innocuous.Endpoint as magic
from innocuous.MagicObj import MagicObj

class Model(nn.Module):
  def __init__(self, img_channel=1, out_channels=10):
    super(Model, self).__init__()
    self.cnn1 = nn.Conv2d(in_channels=img_channel, out_channels=16, kernel_size=5, stride=1, padding=0)
    self.relu1 = nn.ReLU() 
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=2) 
    self.fc1 = nn.Linear(32 * 4 * 4, out_channels) 
  
  def forward(self, x):
    out = self.cnn1(x)
    out = self.relu1(out)
    out = self.maxpool1(out)
    out = self.cnn2(out)
    out = self.relu2(out)
    out = self.maxpool2(out)
    out = out.view(out.size(0), -1)
    out = self.fc1(out)
    return out
    
def main(lr=0.001, epochs=2, batch_size=256):
    mj = MagicObj()
    fileHelper = magic.FileHelper()

    dataset_path = mj.get_path('/Users/noam/Downloads/mnist')


    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(val_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Model()
    pretrained_state = fileHelper.get("data://checkpoint/models/checkpoint.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prefix = 'classifier.'
    loaded_dict = torch.load(pretrained_state, map_location=device)
    adapted_dict = model.state_dict()
    adapted_dict.update(loaded_dict)
    model.load_state_dict(adapted_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        eval_loss = 0.0
        eval_acc = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data in tepoch:
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                checkpoint = model.state_dict()

                _, pred = outputs.max(1)
                num_correct = (pred == labels).sum().item()
                acc = num_correct/images.shape[0]
                tepoch.set_postfix(loss=loss.item(), accuracy=acc)

        model.eval()
        with tqdm(val_loader, unit="batch") as tepoch:
            for data in tepoch:
                images, labels = data
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item()    

                    _, pred = outputs.max(1)
                    num_correct = (pred == labels).sum().item()
                    acc = num_correct/images.shape[0]
                    eval_acc += acc
                    tepoch.set_postfix(loss=loss.item(), accuracy=acc)

        mj.torch_save(checkpoint=checkpoint, path='/Users/noam/Downloads', epoch=epoch)
        mj.log(accuracy=eval_acc/len(val_loader), loss=eval_loss/len(val_loader))