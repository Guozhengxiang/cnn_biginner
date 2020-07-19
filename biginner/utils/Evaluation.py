from utils import config


def label_accuracy_calc(model, test_dataloader):
    model.eval()
    test_acc = 0.0
    for i, (image, label) in enumerate(test_dataloader):
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()

        out = model(image).max(1)[1].cpu().data.numpy()
        test_acc += sum(out == label.cpu().data.numpy())

    return test_acc/len(test_dataloader)


