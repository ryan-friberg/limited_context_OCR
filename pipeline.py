import torch
import pytesseract
from torchvision import transforms
from model import OCRResNet, Extractor, label_map
from datasets import MultiDataSet

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
model_chkpt = 'model_checkpoint/ckpt.pth'
data_dir = 'data/multiple/jpg'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = OCRResNet()
model.to(device)
chkpt = torch.load(model_chkpt)
model.load_state_dict(chkpt['net'])
extractor = Extractor(64, label_map)

# define dataset (no need for test/train)
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Grayscale(1.0), # grayscale every image
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data = MultiDataSet(data_dir, data_transforms)

correct = 0
total = 0
with torch.no_grad():
    for img in iter(data):
        labels, char_imgs = extractor.forward(img)
        char_imgs.to(device)
        for label, char_img in zip(labels, char_imgs):
            output = model(char_img)
            _, predicted_char = torch.max(output,1)
            if predicted_char == label:
                correct += 1
            total += 1
print("OCR Pipeline Accuracy (Compared to Google Tesseract): %.3f" % (correct/total))