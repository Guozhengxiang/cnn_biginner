import torch
from torch import nn
from dataset.Official_dataset import official_set
from model.Model import MyModel
from utils import config, label_accuracy_calc


train_dataloader, test_dataloader = official_set('MNIST', 10,
                                                 is_download=False,
                                                 is_dataloader=True)

model = MyModel()
if config.use_gpu:
    model = model.cuda()


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=config.learning_rate,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)

# train
for epoch in range(config.epoch):
    train_loss = 0
    for step, (image, label) in enumerate(train_dataloader):
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()

        out = model(image)
        loss = loss_function(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    test_acc = label_accuracy_calc(model, test_dataloader)
    print('Epoch:{}, Train_loss:{:.5f}, Test_acc:{:.5f}'.format(
        epoch, train_loss / len(train_dataloader), test_acc))

# save
torch.save(model, 'model.pth')

