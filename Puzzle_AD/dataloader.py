import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset,Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.datasets import ImageFolder
from PIL import Image
import lxml.etree as ET


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, split,transforms):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.segmentation_dir = os.path.join(root_dir, 'SegmentationClass')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.image_ids = []        
        self.transform = transforms
        if split == 'train':
            self.split_list_path = os.path.join(root_dir, 'ImageSets/Segmentation/train.txt')
        elif split == 'val':
            self.split_list_path = os.path.join(root_dir, 'ImageSets/Segmentation/val.txt')
        else:
            raise Exception('Choose between "train" and "val"')
        f = open(self.split_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.image_ids.append(line)
        f.close()
        self.label_mapping = {'aeroplane':0, 'person':1, 'tvmonitor':2, 'dog':3, 'chair':4, 'bird':5, 'bottle':6, 'boat':7, 
                 'diningtable':8, 'train':9, 'motorbike':10, 'horse':11, 'cow':12, 'bicycle':13, 'car':14, 'cat':15, 
                 'sofa':16, 'bus':17, 'pottedplant':18, 'sheep':19}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, filename + '.jpg')
        annotation_path = os.path.join(self.annotation_dir, filename + '.xml')        
        image = Image.open(image_path).convert('RGB').resize((224,224)) # Read image 
        annotation = self._parse_annotation(annotation_path) # Read label 
        if self.transform is not None:
            image = self.transform(image)
        return image,annotation

    def _parse_annotation(self, annotation_path:str) -> list:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        labels = []
        for obj in root.findall('object'):
            label = obj.find('name').text 
            labels.append(label)
        main_label = labels[0]
        number_label = self.label_mapping.get(main_label)
        return number_label


class vocdata(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        input_data , _ = self.dataset[index]
        return input_data

    def __len__(self):
        return len(self.dataset)


def load_data(config):
    normal_class = config['normal_class']
    batch_size = config['batch_size']
    img_size = config['image_size']

    if config['dataset_name'] in ['cifar10']:
        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['mnist']:
        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['fashionmnist']:
        img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]

        train_set, val_set = torch.utils.data.random_split(dataset, [dataset.data.shape[0] - 851, 851])

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)

    elif config['dataset_name'] in ['brain_tumor', 'head_ct']:
        img_transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        root_path = 'Dataset/medical/' + config['dataset_name']
        train_data_path = root_path + '/train'
        test_data_path = root_path + '/test'
        dataset = ImageFolder(root=train_data_path, transform=img_transform)
        load_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataset_array = next(iter(load_dataset))[0]
        my_dataset = TensorDataset(train_dataset_array)
        train_set, val_set = torch.utils.data.random_split(my_dataset, [train_dataset_array.shape[0] - 5, 5])

        test_set = ImageFolder(root=test_data_path, transform=img_transform)

    elif config['dataset_name'] in ['coil100']:
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        root_path = 'Dataset/coil100/' + config['dataset_name']
        train_data_path = root_path + '/train'
        test_data_path = root_path + '/test'
        dataset = ImageFolder(root=train_data_path, transform=img_transform)
        load_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataset_array = next(iter(load_dataset))[0]
        my_dataset = TensorDataset(train_dataset_array)
        train_set, val_set = torch.utils.data.random_split(my_dataset, [train_dataset_array.shape[0] - 5, 5])

        test_set = ImageFolder(root=test_data_path, transform=img_transform)

    elif config['dataset_name'] in ['MVTec']:
        data_path = 'Dataset/MVTec/' + normal_class + '/train'
        data_list = []

        orig_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

        orig_dataset = ImageFolder(root=data_path, transform=orig_transform)

        train_orig, val_set = torch.utils.data.random_split(orig_dataset, [len(orig_dataset) - 25, 25])
        data_list.append(train_orig)

        for i in range(3):
            img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomAffine(0, scale=(1.05, 1.2)),
                transforms.ToTensor()])

            dataset = ImageFolder(root=data_path, transform=img_transform)
            data_list.append(dataset)

        dataset = ConcatDataset(data_list)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=800, shuffle=True)
        train_dataset_array = next(iter(train_loader))[0]
        train_set = TensorDataset(train_dataset_array)

        test_data_path = 'Dataset/MVTec/' + normal_class + '/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    elif config['dataset_name'] in ['pvoc']:
        data_path = './VOC2012/'
        img_transform = transforms.Compose([transforms.ToTensor()])

        train_orig = PascalVOCDataset(data_path, 'train', transforms=img_transform)
        original_trainset, validation_set = torch.utils.data.random_split(train_orig, [len(train_orig) - 25, 25])

        test_set = PascalVOCDataset(data_path, 'val', transforms=img_transform)   

        subset_label = config['normal_class']

        # Iterate over the dataset and filter samples with the subset_label
        train_subset_indices = []
        for idx in range(len(original_trainset)):
            _, label = original_trainset[idx]
            if subset_label == label:
                train_subset_indices.append(idx)

        # Create a Subset of the original dataset using the subset_indices
        train_subset = torch.utils.data.Subset(original_trainset, train_subset_indices)
        train_set = vocdata(train_subset) 
        val_set = vocdata(validation_set)


    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
    )

    return train_dataloader, val_dataloader, test_dataloader
