import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import *
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.models import vgg16

random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
if not os.path.exists('vgg_models'):
    os.mkdir('vgg_models')


class CUHK_PEDES(Dataset):
    def __init__(self, dataset, image_size=(256, 256), images_dir='./data/CUHK-PEDES/imgs'):
        self.images_dir = images_dir
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        item = self.dataset[index]
        file_path = item['file_path']

        pid = int(item['id']) - 1
        image_path = os.path.join(self.images_dir, file_path)
        image = Image.open(image_path)
        # resize image to 256x256
        image = self.transform(image)
        return image, pid

    def __len__(self):
        return len(self.dataset)


with open('./data/reid_raw.json', 'r') as f:
    data_json = json.load(f)
random.shuffle(data_json)
train_set = data_json[:-100]
test_set = data_json[-100:]
train_set = CUHK_PEDES(train_set)
test_set = CUHK_PEDES(test_set)
print(len(train_set))
batch_size = 64
num_worker = 10
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
model = vgg16(pretrained=True)
model.classifier[-1] = nn.Linear(4096, 13003)

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
epochs = 100
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(params=model.parameters(), lr=1e-5)
lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
# train

for e in range(epochs):
    model.train()
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_func(outputs, labels)
        y_pred = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y_pred.detach().cpu().numpy(), labels.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            end_time = time.time()
            sum_time = (end_time - start_time) / 50
            start_time = time.time()
            rest_batch = len(train_loader) - i - 1
            sum_time = sum_time * ((epochs - e - 1) * len(train_loader) + rest_batch)
            print(f'Epoch {e}/{epochs}, Batch: {i}/{len(train_loader)}, '
                  f'loss: {loss.item():.4f}, acc: {acc * 100:.2f}%, rest:{sum_time / 3600:.2f}h')

    sum_correct = 0
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = loss_func(outputs, labels)
            y_pred = torch.argmax(outputs, dim=1)
            n_correct = accuracy_score(y_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), normalize=False)
            sum_correct += n_correct
    print(f'Test {e}/{epochs}, Batch: {i}/{len(train_loader)}, acc: {sum_correct / len(test_set) * 100}%')
    torch.save(model.state_dict(), f'vgg_models/epoch_{e}.pkl')
    print(f'epoch {e} saved!')
    lr_scheduler.step()
