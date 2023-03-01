import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import argparse
from torchvision.io import read_image
# import torchsummary
from model import label_map

from datasets import OCRDataSet, collate_fn
from model import OCRResNet, trainOCR, testOCR

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch OCR Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
parser.add_argument('--optim', default='sgd', type=str, help='training optimizer: sgd, sgd_nest, adagrad, adadelta, or adam')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--train_split', default=0.75, type=float, help='percentage of dataset to be used for training')
parser.add_argument('--epochs', default=5, type=int, help='numper of training epochs')
parser.add_argument('--load', default='', type=str, help='load model checkpoint')

def main():
    # multi_path = "data/multiple"
    single_path = "data/single/Img"
    args = parser.parse_args()

    data_transforms = transforms.Compose([transforms.ToTensor(),
                                        #   transforms.Resize(3,64,64),
                                        #   transforms.Grayscale(0.4),
                                          transforms.RandomAdjustSharpness(1.5, 0.3),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # define ResNet model
    OCR_model = OCRResNet()
    # torchsummary.summary(OCR_model, (3, 64, 64))

    # check if requested GPU is available
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
    
    # load a pre-trained if need be
    start = 0
    if args.load != '': # model needs to be loaded to the same device it was saved from
        print('==> Loading checkpoint...')
        checkpoint = torch.load(args.load)
        OCR_model.load_state_dict(checkpoint['model'])
        acc = checkpoint['acc'] # ended up being unused
        start = checkpoint['epoch'] + 1

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # pick the model from the arguments
    if args.optim == 'sgd':
        print("==> SGD")
        optim = torch.optim.SGD(OCR_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) # values suggested from another class' assignment
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
    for epoch in range(start, start+args.epochs):
        trainOCR(OCR_model, epoch, train_loader, optim, device, criterion)
        testOCR(OCR_model, epoch, test_loader, device, criterion, args.optim)


if __name__ == '__main__':
    main()