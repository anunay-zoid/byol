import os

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = ['.jpg' , '.png' , 'jpeg']

print(torch.__version__)
torch.manual_seed(0)

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform1 = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])
        self.transform2 = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale),
            transforms.ColorJitter(0.8 , 0.8 , 0.8 , 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur((3, 3), (1.0, 2.0)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform1(img) , self.transform2(img)
def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    # data_transform = get_simclr_data_transforms(**config['data_transforms'])

    # train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train+unlabeled', download=True,
                                #    transform=MultiViewDataInjector([data_transform, data_transform]))

    train_dataset = ImagesDataset("/home/anunay/Documents/crop_pad" , 256)

    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
