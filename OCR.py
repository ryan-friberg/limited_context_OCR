import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
import os
import glob
import itertools
import argparse
from torchvision.io import read_image
import pytesseract
import torchsummary

from datasets import OCRDataSet, collate_fn
from model import OCRResNet, trainOCR, testOCR

cudnn.benchmark = True
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

parser = argparse.ArgumentParser(description='PyTorch OCR Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
parser.add_argument('--optim', default='sgd', type=str, help='training optimizer: sgd, sgd_nest, adagrad, adadelta, or adam')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--train_split', default=0.75, type=float, help='percentage of dataset to be used for training')
# parser.add_argument('--data_path', default='./data', type=str, help='location of data')
parser.add_argument('--epochs', default=5, type=int, help='numper of training epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

def main():
    # multi_path = "data/multiple"
    single_path = "data/single/Img"

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

    args = parser.parse_args()

    # convert training data to a tensor, model, + TODO ADD MORE TRANSFROMS
    # RandomGrayscale([p])
    # RandomAdjustSharpness(sharpness_factor[, p])

    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(3,64,64),
                                          transforms.Grayscale(0.4),
                                          transforms.RandomAdjustSharpness(1.5, 0.3),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # define ResNet model
    OCR_model = OCRResNet()
    torchsummary.summary(OCR_model, (3, 64, 64))
    # determine which device to run the model on
    print("==> Configuring device...")
    if args.device == 'cuda':
        print("Checking for GPU...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            OCR_model = torch.nn.DataParallel(OCR_model)
            torch.backends.cudnn.benchmark = True
            print("GPU found, using GPU...")
        else:
            print("GPU not found, using CPU...")
    else:
        print("Using CPU...")
        device = 'cpu'
    OCR_model.to(device)
    
    # define loss function
    criterion = nn.CrossEntropyLoss()

    # pick the model from the arguments
    if args.optim == 'sgd':
        print("==> SGD")
        optim = torch.optim.SGD(OCR_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.optim == 'sgd_nest':
        print("==> SGD w/ Nesterov")
        optim = torch.optim.SGD(OCR_model.parameters(), nesterov=True, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.optim == 'adam':
        print("==> Adam")     
        optim = torch.optim.Adam(OCR_model.parameters(), lr=args.lr, weight_decay=5e-4)
    if args.optim == 'adagrad':
        print("==> Adagrad")
        optim = torch.optim.Adagrad(OCR_model.parameters(), lr=args.lr, weight_decay=5e-4)
    if args.optim == 'adadelta':
        print("==> Adadelta")
        optim = torch.optim.Adadelta(OCR_model.parameters(), lr=args.lr, weight_decay=5e-4)

    # build datasets and split into training/testing
    print("==> Building datasets")
    character_dataset = OCRDataSet(single_path, data_transforms, label_map)
    train_size = int(args.train_split * len(character_dataset))
    test_size = len(character_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(character_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=4,
                                               shuffle=True, 
                                               num_workers=args.num_workers, 
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=4,
                                               shuffle=False, 
                                               num_workers=args.num_workers, 
                                               collate_fn=collate_fn)
    
    # train the model
    print("Beginning Training...")
    best_acc = 0
    for epoch in range(args.epochs):
        trainOCR(OCR_model, epoch, train_loader, optim, device, criterion)
        testOCR(OCR_model, epoch, test_loader, best_acc, device, criterion)
        optim.step()


if __name__ == '__main__':
    main()