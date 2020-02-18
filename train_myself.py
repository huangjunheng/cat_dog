import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet50
from utils import Flatten
from cnnNet import ResNet


batchsz = 128
epochs = 50
lr = 2e-3

data_transform = transforms.Compose([
    transforms.Resize(84),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='./training_set/',
                                     transform=data_transform)

train_iters = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)

test_dataset = datasets.ImageFolder(root='./test_set/',
                                     transform=data_transform)
test_iters = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)


device = torch.device('cuda:0')

# train_model = resnet50(pretrained=True)
# net = nn.Sequential(
#     *list(train_model.children())[:-1],
#     Flatten(),
#     nn.Linear(2048, 2)
# ).to(device)

net = ResNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_id, (x, y) in enumerate(train_iters):

        x, y = x.to(device), y.to(device)

        logits = net(x)

        loss = criteon(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            print('epoch:[{}/{}], loss:{:.4f}'.format(epoch, epochs, loss))

net.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for x, y in test_iters:
        x = x.to(device)
        y = y.to(device)
        # x = x.view(x.size(0), 28 * 28)

        logits = net(x)

        pred = logits.argmax(dim=1)

        total += len(y)

        correct += pred.eq(y).sum().float().item()

print('acc:{:.4f}'.format(correct / total))


# acc:0.8220