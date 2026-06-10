import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader


# 加载数据 
dataset = CIFAR10("./cifar10", train=True, transform=T.ToTensor())
dataloder = DataLoader(dataset, batch_size=16, shuffle=True)
# Get a batch of 16 images
images, labels = next(iter(dataloder))

# Create a 4x4 grid plot
fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    # Get the i-th image from the batch
    img = images[i].numpy().transpose(1, 2, 0)
    
    # Display the image
    ax.imshow(img)
    
    # Remove axis ticks for cleaner look
    ax.axis('off')
    
    # Optional: Add label as title
    ax.set_title(f'Label: {labels[i]}', fontsize=8)

plt.tight_layout()
plt.savefig('output_grid.png')
plt.show()
plt.close()
def get_model(output_dim=10):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# memory bank
C = 1024 #模型输出维度
N = dataloder.batch_size
K = 4096


f_q = get_model(C).to(device) # 模型输出 C 维度
f_k = get_model(C).to(device) 
f_k.load_state_dict(f_q.state_dict())
queue = torch.randn(C, K).to(device)

queue_ptr = 0

W = 0.99

optimizer = torch.optim.Adam(f_q.parameters(), lr=0.001)
#  info_nce loss
def info_nce_loss(q, k, queue,temperature = 0.07):
    q = F.normalize(q, dim=1,p=2) # (batchsize,C)
    k = F.normalize(k, dim=1,p=2) # (batchsize,C)
    queue = F.normalize(queue, dim=0, p=2)  # (C,4096)
    positive_simlarity = torch.bmm(q.view(N,1,C), k.view(N,C,1)) #(batchsize,batchsize)
    negative_simlarity = torch.mm(q, queue)
    simlarity = torch.cat([positive_simlarity.squeeze(-1), negative_simlarity], dim=1) #(batchsize, K+1)
    labels = torch.zeros(N).to(device).long()
    loss = nn.CrossEntropyLoss()(simlarity / temperature, labels)
    return loss
# 数据增强
def aug(x):
    return x * 0.01 + torch.randn_like(x) 

for x , _ in dataloder:
    x = aug(x).to(device)
    x_k = aug(x).to(device)
    x_q = aug(x).to(device) 
    q = f_q(x_q)
    k = f_k(x_k)
    k = k.detach()
    loss = info_nce_loss(q, k, queue)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        for param_q, param_k in zip(f_q.parameters(), f_k.parameters()):
            param_k.data = W * param_k.data + (1-W) * param_q.data
    queue[:, queue_ptr:queue_ptr+N] = k.T
    queue_ptr = (queue_ptr + N) % K
    print(loss.item())
    
    