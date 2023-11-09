from test import *
from utils.utils import *
from dataloader import *
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test
from loss_functions import *
from argparse import ArgumentParser
from models.network import get_networks
from pvoc import PascalVOCDataModule

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='/home/napostol/Semantic-Anomaly-Segmentation-Benchmark/KDAD/configs/config.yaml', help="training configuration")


def train(config, train_dataloader, test_dataloader, normal_class):

    direction_loss_only = config["direction_loss_only"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if continue_train:
        vgg, model = get_networks(config, load_checkpoint=True)
    else:
        vgg, model = get_networks(config)

    # Criteria And Optimizers
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    vgg = vgg.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            images = data.to(device)
            output_pred = model.forward(images)
            output_real = vgg(images)
            total_loss = criterion(output_pred, output_real)
            epoch_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    print(f"Epoch {epoch+1} Loss: {epoch_loss}")

    roc_auc = detection_test(model, vgg, test_dataloader, normal_class)
    print(f"Normal Class: {normal_class} Test AUC: {roc_auc}")

    return roc_auc


def main():
    test_aucs = []
    for normal_class in range(20):
        print(f"Normal Class : {normal_class}")
        args = parser.parse_args()
        config = get_config(args.config)

        train_transform= transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
            ])
        normal_classes = [normal_class]
        mod = PascalVOCDataModule(batch_size=32, train_transform=train_transform, val_transform=train_transform, test_transform=train_transform, normal_classes = normal_classes)
        train_dataloader = mod.get_train_dataloader()
        test_dataloader = mod.get_test_dataloader()

        test_auc = train(config, train_dataloader, test_dataloader, normal_class)
        test_aucs.append(test_auc)

    print(f"Test AUCs: {test_aucs}")
    print(f"Average Test AUC: {sum(test_aucs)/len(test_aucs)}")

if __name__ == '__main__':
    main()
