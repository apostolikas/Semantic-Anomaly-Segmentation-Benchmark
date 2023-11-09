import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader,Dataset
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2, BN_layer, AttnBottleneck
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset,PascalVOCDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test, my_eval, evaluate_model
from torch.nn import functional as F
from torchvision import transforms
from pvoc_data import PascalVOCDataModule

class trainvocdata(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        input_data, _ , _ = self.dataset[index]
        return input_data

    def __len__(self):
        return len(self.dataset)
    
class testvocdata(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        input_data, _ , gt = self.dataset[index]
        return input_data, gt

    def __len__(self):
        return len(self.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

# Ignore this train function

#def train(_class_):
    print(_class_)
    epochs = 100
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    orig_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    train_dataset = PascalVOCDataset('./VOC2012', 'train', transforms=orig_transform, )
    test_dataset = PascalVOCDataset('./VOC2012', 'val', transforms=orig_transform, )

    subset_label = _class_

    train_subset_indices = []
    for idx in range(len(train_dataset)):
        _, label, _ = train_dataset[idx]
        if subset_label == label:
            train_subset_indices.append(idx)

    test_subset_indices = []
    for idx in range(len(test_dataset)):
        _, label, _ = test_dataset[idx]
        if subset_label == label:
            test_subset_indices.append(idx)

    # Create a Subset of the original dataset using the subset_indices
    trainset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    testset = torch.utils.data.Subset(test_dataset, test_subset_indices)
    train_set = trainvocdata(trainset) # -> only img
    test_set = testvocdata(testset) # -> img - gt pair
    #ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=lambda x: x)


    encoder, bn = resnet50(pretrained=True)
    bn = BN_layer(AttnBottleneck,3)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    #decoder = de_wide_resnet50_2(pretrained=False)
    decoder = de_resnet50(pretrained=False)
    decoder = decoder.to(device)
    ckp_path = './checkpoints/' + 'resnet34_'+_class_+'.pth'

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img in train_dataloader:
            img = torch.stack(img)
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 10 == 0:
            #auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            auroc_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}'.format(auroc_px))
            #print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            # torch.save({'bn': bn.state_dict(),
            #             'decoder': decoder.state_dict()}, ckp_path)
    #return auroc_px, auroc_sp, aupro_px
    return auroc_px, encoder, bn, decoder


#def train(train_dataloader, val_loader):
    epochs = 100
    learning_rate = 0.005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder, bn = resnet34(pretrained=True) #choose r50 for dino + comment resnet.py line 192 
    # encoder, bn = dino(pretrained=True) # check if it works
    # decoder, _ = dino(pretrained =False) # check if it works
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet34(pretrained=False)
    decoder = decoder.to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))
    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for batch in train_dataloader:
            img = batch[0]
            img = img.repeat(1,3,1,1) # only for mnist 1x32x32
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 10 == 0:
            auroc_px = my_eval(encoder, bn, decoder, val_loader, device)
            print('Val Pixel Auroc:{:.3f}'.format(auroc_px))
    return auroc_px, encoder, bn, decoder


def train(train_dataloader, _class_):
    print(f"Normal Class : {_class_}")
    epochs = 100
    learning_rate = 0.005
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder, bn = resnet50(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet50(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

    return encoder, bn, decoder



if __name__ == "__main__":

    test_aucs = []

    for _class_ in range(20):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_transform= transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        normal_classes = [_class_]
        mod = PascalVOCDataModule(batch_size=16, train_transform=train_transform, val_transform=train_transform, test_transform=train_transform, normal_classes = normal_classes)
        train_dataloader = mod.get_train_dataloader()
        test_dataloader = mod.get_test_dataloader()

        encoder, bn, decoder = train(train_dataloader, _class_)

        test_auc = my_eval(encoder, bn, decoder, test_dataloader, device)
        print(f"Test pixel auroc {test_auc}")
        test_aucs.append(test_auc)

    print(f"Test AUCs {test_aucs}")
    print(f"Mean AUC {np.mean(test_aucs)}")
