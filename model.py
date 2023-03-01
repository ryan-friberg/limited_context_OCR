import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

label_map = {1: "0", 2: "1", 3: "2", 4: "3", 5: "4",
                6: "5", 7: "6", 8: "7", 9: "8", 10: "9",
                11: "A", 12: "B", 13: "C", 14: "D", 15: "E",
                16: "F", 17: "G", 18: "H", 19: "I", 20: "J",
                21: "K", 22: "L", 23: "M", 24: "N", 25: "O",
                26: "P", 27: "Q", 28: "R", 29: "S", 30: "T",
                31: "U", 32: "V", 33: "W", 34: "X", 35: "Y",
                36: "Z", 37: "a", 38: "b", 39: "c", 40: "d",
                41: "e", 42: "f", 43: "g", 44: "h", 45: "i",
                46: "j", 47: "k", 48: "l", 49: "m", 50: "n",
                51: "o", 52: "p", 53: "q", 54: "r", 55: "s",
                56: "t", 57: "u", 58: "v", 59: "w", 60: "x",
                61: "y", 62: "z"}

class Extractor():
    def __init__(self, label_map, out_shape):
        self.label_map = label_map
        self.out_shape = out_shape

    # use Google Tesseract to extrapolate bounding boxes around each character in an image
    # Tesseract also supplies its guess for the character in the box, which is used as a pseudo-label
    # system within the OCR pipeline
    def forward(self, img_name):
        labels, imgs = [], []
        img = Image.open(img_name)
        wImg, hImg = img.size
        boxes = pytesseract.image_to_boxes(img)
        if len(boxes) == 0:
            return
        
        # extract the coordinates of the each bounding box found in the image
        for box in boxes.splitlines():
            vals = box.split(' ')
            if vals[0] not in self.label_map.values():
                continue
            labels.append(vals[0])
            x, y, w, h = int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])
            cropped = img.crop((x, hImg-h, w, hImg-y))
            w2, h2 = cropped.size

            # add some white pixel padding around the image
            padded = Image.new(cropped.mode, (w2+10, h2+10), (255, 255, 255))
            padded.paste(cropped, (5,5))

            # match the output shape to the expected input shape of the OCRResNet
            char_img = padded.resize((self.out_shape, self.out_shape))
            imgs.append(char_img)
        return labels, imgs

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
    
# Simple ResNet model based off of the paper Deep Residual Learning for Image Recognition
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
    print("Beginning training epoch", epoch)
    train_loss = 0
    correct = 0
    total = 0
    start = time.perf_counter()
    model.train()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += torch.sum(preds == labels.data)
    end = time.perf_counter()
    print("Training", epoch, "time: %.3f" % (end-start), 'Loss: %.3f Acc: %.3f (%d/%d)' % 
          (train_loss/(batch_idx+1), correct/total, correct, total))

def testOCR(model, epoch, testloader, device, criterion, optim_name):
    print("\nTesting...") 
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

        print('Test Loss: %.3f Acc: %.3f (%d/%d)' % (test_loss/(batch_idx+1), correct/total, correct, total))
        print('Saving model...')
        state = {'model': model.state_dict(), 'acc': 100.*correct/total, 'epoch': epoch}
        if not os.path.isdir('model_checkpoint'):
            os.mkdir('model_checkpoint')
        checkpoint_name = optim_name + 'chpkt' + str(epoch) + '.pth'
        torch.save(state, checkpoint_name)