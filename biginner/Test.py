import torch
import torchvision

from dataset.Official_dataset import official_set
from utils import label_accuracy_calc


model = torch.load('model.pth')

_, test_dataloader = official_set('MNIST', 1, is_download=False, is_dataloader=True)

test_acc = label_accuracy_calc(model, test_dataloader)
print('Test_acc:{:.5f}'.format(test_acc))
