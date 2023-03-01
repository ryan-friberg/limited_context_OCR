import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

## simple implementation of a residual network
class Block(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(Block, self).__init__()
        self.inp = input_channels
        self.outp = output_channels
        self.layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(output_channels), 
                                    nn.ReLU(),
                                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(output_channels))
        self.downsample_layer = nn.Sequential()
        if (stride != 1) or (input_channels != output_channels):
            self.downsample_layer = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0),
                                                  nn.BatchNorm2d(output_channels))

    def forward(self, input):
        return F.relu(self.layers(input) + self.downsample_layer(input))

class OCRResNet(nn.Module):
    def __init__(self, num_classes=62):
        super(OCRResNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())
        self.block1 = nn.Sequential(Block(32, 64, 2),
                                    Block(64, 64, 1))
        self.block2 = nn.Sequential(Block(64, 128, 1),
                                    Block(128, 128, 1))
        self.block3 = nn.Sequential(Block(128, 256, 2),
                                    Block(256, 256, 1))
        self.block3 = nn.Sequential(Block(128, 512, 2))
        self.pool   = nn.AvgPool2d(8)
        self.linear = nn.Linear(4096, num_classes)
        
        self.layers = nn.Sequential(self.conv, self.block1, self.block2, 
                                    self.block3, self.pool)

    def forward(self, input):
        output = self.layers(input)
        return output.view(output.size(0), -1)
        
    
def trainOCR(model, epoch, trainloader, optim, device, criterion):
    print("Beginning traing epoch", epoch)
    train_loss = 0
    correct = 0
    total = 0
    start = time.perf_counter()
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    end = time.perf_counter()
    # log = "Training Epoch: %d - Runtime: %.3f - Loss: %.3f - Acc: %.3f  - Precision %.3f - Recall %.3f - F1 %.3f"
    # metrics = "TP: %d - TN %d - FP %d - FN"
    print("Training", epoch, "time: %.3f" % (end-start), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % 
          (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def testOCR(model, epoch, testloader, best_acc, device, criterion):
    print("\nTesting...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {'model': model.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('model_checkpoint'):
            os.mkdir('model_checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc