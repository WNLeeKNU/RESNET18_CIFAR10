import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import os
import torchvision.models as models
if __name__ == '__main__':

    # Simple Learning Rate Scheduler
    def lr_scheduler(optimizer, epoch):
        lr = learning_rate
        if epoch >= 50:
            lr /= 1
        if epoch >= 100:
            lr /= 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Xavier
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)



    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 중에 택일하여 사용

    # model.apply(init_weights)
    model = model.to(device)

    learning_rate = 0.01
    num_epoch = 150
    model_name = 'model.pth'

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_loss = 0
    valid_loss = 0
    correct = 0
    total_cnt = 0
    best_acc = 0

    # Train
    for epoch in range(num_epoch):
        print(f"====== {epoch + 1} epoch of {num_epoch} ======")
        model.train()
        lr_scheduler(optimizer, epoch)
        train_loss = 0
        valid_loss = 0
        correct = 0
        total_cnt = 0
        # Train Phase
        for step, batch in enumerate(train_loader):
            #  input and target
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            logits = model(batch[0])
            loss = loss_fn(logits, batch[1])
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predict = logits.max(1)

            total_cnt += batch[1].size(0)
            correct += predict.eq(batch[1]).sum().item()

            if step % 100 == 0 and step != 0:
                print(f"\n====== {step} Step of {len(train_loader)} ======")
                print(f"Train Acc : {correct / total_cnt}")
                print(f"Train Loss : {loss.item() / batch[1].size(0)}")

        correct = 0
        total_cnt = 0

        # Test Phase
        with torch.no_grad():
            model.eval()
            for step, batch in enumerate(test_loader):
                # input and target
                batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
                total_cnt += batch[1].size(0)
                logits = model(batch[0])
                valid_loss += loss_fn(logits, batch[1])
                _, predict = logits.max(1)
                correct += predict.eq(batch[1]).sum().item()
            valid_acc = correct / total_cnt
            print(f"\nValid Acc : {valid_acc}")
            print(f"Valid Loss : {valid_loss / total_cnt}")

        if (valid_acc > best_acc):
            best_acc = valid_acc
            torch.save(model, model_name)
            print("Model Saved!")