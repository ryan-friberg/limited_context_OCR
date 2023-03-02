import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import itertools

### OCR Custom dataset
class OCRDataSet(Dataset):
    def __init__(self, images_dir, transforms, label_map):
        self.label_map = label_map
        self.image_transforms = transforms
        self.classes = self.get_class_names(images_dir)
        self.class_labels = {name: idx for idx, name in enumerate(self.classes)}

        image_files, labels = self.get_image_filenames_with_labels(
            images_dir,
            self.classes,
            self.class_labels,
        )
        
        self.image_files = np.array(image_files)
        self.labels = np.array(labels).astype("int")
        self.num_images = len(self.image_files)
        
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_files[idx]).convert('RGB')
            if image.size != (64,64):
                image = image.resize((64,64)) 
            label = self.labels[idx]
            image = self.image_transforms(image)
            return image, label
        except:
            return None
        
    def __len__(self):
        return self.num_images
    
    def get_class_names(self, images_dir):
        class_name_dirs = glob.glob(images_dir + "/*")
        class_names = [self.label_map[int(name.replace(images_dir + "/", ""))] for name in class_name_dirs]
        return sorted(class_names)
    
    def get_image_filenames_with_labels(self, images_dir, class_names, class_labels):
        image_files = []
        labels = []
        supported_file_types = ["/*.png"]
        
        files = os.listdir(images_dir)
        for name in files:
            if name == ".DS_Store":
                continue
            image_class_dir = os.path.join(images_dir, name)
            image_class_files = list(itertools.chain.from_iterable(
                [glob.glob(image_class_dir + file_type) for file_type in supported_file_types]))
            image_files += image_class_files
            labels += [int(name)] * len(image_class_files)
        return image_files, labels


class MultiDataSet():
    def __init__(self, images_dir):
        image_files = self.get_image_filenames_with_labels(images_dir)       
        self.image_files = np.array(image_files)
        self.num_images = len(self.image_files)
        
    def __getitem__(self, idx):
        try:
            # print(self.image_files[idx])
            img = Image.open(self.image_files[idx]).convert('RGB')
            # print(img.size)
            return img
        except:
            return None
        
    def __len__(self):
        return self.num_images
    
    def get_image_filenames_with_labels(self, images_dir):
        image_files = []
        files = os.listdir(images_dir)
        for name in files:
            if name == ".DS_Store":
                continue
            image_files.append(images_dir + '/' + name)
        return image_files


def collate_fn(batch):
    # Filter failed images first
    batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    images = torch.stack([b[0] for b in batch])
    labels = torch.LongTensor([b[1] for b in batch])
    
    return images, labels