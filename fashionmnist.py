import torch.utils.data
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5) # n*16*24*24
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2) # n*16*12*12
        self.conv2 = nn.Conv2d(16, 32, 5) # n*32*8*8
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2) # n*32*4*4
        self.linear1 = nn.Linear(32 * 4 * 4, 128)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(128, 10) # n*10
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.linear2(self.relu3(self.linear1(x)))
        return x


def validation(data_loader, model: nn.Module, loss):
    model.eval()
    ll = 0.0
    length = 0
    acc = 0
    for x_test, y_test in data_loader:
        y_pred = model(x_test)
        ll += (loss(y_pred, y_test) * x_test.size(dim=0)).item()
        length += x_test.size(dim=0)
        acc += (y_pred.argmax(dim=1) == y_test).sum().item()
    model.train()
    return ll/length, acc/length


def main():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = datasets.FashionMNIST('data', train=True, transform=trans, download=True)
    test_data = datasets.FashionMNIST('data', train=False, transform=trans, download=True)

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=64,
                                                    shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=64)
    model = Model()
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(1000):
        for i, (x_train, y_train) in enumerate(train_data_loader):
            y_pred = model(x_train)
            ll = loss(y_pred, y_train)
            optim.zero_grad()
            ll.backward() # weight_grad
            optim.step() # weight = weight - weight_grad*lr
            losses.append(ll.item())
            if i % 100 == 0:
                print(f'epoch:{epoch:4d}, iter:{i:4d}, loss:{sum(losses)/len(losses)}')
                losses = []
        test_loss, acc = validation(test_data_loader, model, loss)
        print(f'test_loss:{test_loss:.4f}, acc:{acc*100:.2f}%')


if __name__ == '__main__':
    main()