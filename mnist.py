import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],
                                                     std=[0.5])])

data_train = datasets.MNIST(root='./data/', transform=transform, train=True, download=True)
data_test = datasets.MNIST(root='./data/', transform=transform, train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)


  def forward(self, x):
    # Max pooling over a (2, 2) window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # If the size is a square, you can specify with a single number
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 5
# model.load_state_dict(torch.load('model_parameter.pkl'))

for epoch in range(n_epochs):
  running_loss = 0.0
  running_correct = 0
  print('Epoch {}/{}'.format(epoch, n_epochs))
  print('-' * 10)
  for data in data_loader_train:
    X_train, y_train = data
    X_train, y_train = Variable(X_train), Variable(y_train)
    outputs = model(X_train)
    _, pred = torch.max(outputs.data, 1)
    optimizer.zero_grad()
    loss = cost(outputs, y_train)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    running_correct += torch.sum(pred == y_train.data)

  testing_correct = 0
  for data in data_loader_test:
    X_test, y_test = data
    X_test, y_test = Variable(X_test), Variable(y_test)
    outputs = model(X_test)
    _, pred = torch.max(outputs.data, 1)
    testing_correct += torch.sum(pred == y_test.data)

  print('Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}'.format(
          running_loss / len(data_train), 100 * running_correct / len(data_train),
          100 * testing_correct / len(data_test)))

# torch.save(model.state_dict(), 'model_parameter.pkl')

data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=4,
                                               shuffle=True)
X_test, y_test = next(iter(data_loader_test))
inputs = Variable(X_test)
pred = model(inputs)
_, pred = torch.max(pred, 1)

print('Predict Label is:', [i for i in pred.data])
print('Real Label is:', [i for i in y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1, 2, 0)

std = [0.5]
mean = [0.5]
img = img * std + mean
plt.imshow(img)
plt.show()



