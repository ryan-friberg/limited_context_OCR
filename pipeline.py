import numpy as np
import torch
import pytesseract
from torchvision import transforms
from model import OCRResNet, Extractor, label_map
from datasets import OCRDataSet, MultiDataSet, collate_fn

import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
model_chkpt = 'model_checkpoint/redo_sgdchpkt4.pth'
data_dir = 'data/multiple/jpg'
single_path = "data/single/Img"
batch_size = 4

label_map_inverted = {"0":0, "1":1, "2":2, "3":3, "4":4,
                     "5":5, "6":6, "7":7, "8":8, "9":9,
                     "A":10, "B":11, "C":12, "D":13, "E":14,
                     "F":15, "G":16, "H":17, "I":18, "J":19,
                     "K":20, "L":21, "M":22, "N":23, "O":24,
                     "P":25, "Q":26, "R":27, "S":28, "T":29,
                     "U":30, "V":31, "W":32, "X":33, "Y":34,
                     "Z":35, "a":36, "b":37, "c":38, "d":39,
                     "e":40, "f":41, "g":42, "h":43, "i":44,
                     "j":45, "k":46, "l":47, "m":48, "n":49,
                     "o":50, "p":51, "q":52, "r":53, "s":54,
                     "t":55, "u":56, "v":57, "w":58, "x":59,
                     "y":60, "z":61}

print("==> Building ResNet...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("==> Using " + device + "...")
model = OCRResNet()
model = torch.nn.DataParallel(model)
model.to(device)
chkpt = torch.load(model_chkpt)
model.load_state_dict(chkpt['model'])

# define dataset (no need for test/train)
print("==> Building dataset...")
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Grayscale(num_output_channels=3),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

extractor = Extractor(64, label_map, data_transforms)
data = MultiDataSet(data_dir)
print(len(data))

def model_pass(batch, labels):
    correct = 0
    total = 0
    for i in range(len(labels)):
        labels[i] = label_map_inverted[labels[i]]
        
    labels = torch.tensor(labels)
    batch = batch.cuda()
    output = model(batch)
    output = output.cpu()
    _, predicted_chars = output.max(1) # torch.max(output,1)
    total += len(labels)
    correct += predicted_chars.eq(labels).sum().item()
    return correct, total

correct = 0
total = 0
with torch.no_grad():
    batch = []
    batch_labels = []
    i = 0
    for img in iter(data):
        if (i % 500) == 0 and total != 0:
            print(i, "acc:", str(correct/total))
        labels, char_imgs = extractor.forward(img)
        for label, char_img in zip(labels, char_imgs):
            batch.append(char_img)
            batch_labels.append(label)
            if len(batch) == batch_size:
                # collate on-the-fly
                batch = torch.stack([b for b in batch])
                c, t = model_pass(batch, batch_labels)
                correct += c
                total += t
                batch = []
                batch_labels = []
        i += 1
print("OCR Pipeline Accuracy (Compared to Google Tesseract): %.3f" % (correct/total))