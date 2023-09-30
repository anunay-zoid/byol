from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

image_size = 256

def expand_greyscale(t):
    return t.expand(3, -1, -1)

def regression_loss(x, y):
        # x = F.normalize(x, dim=1)
        # y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

def image_to_tensor(img_path):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])
        img = Image.open(img_path)
        img = img.convert('RGB')

        return transform(img)

PATH = "/home/zoid/PyTorch-BYOL/runs/Sep30_17-22-36_zoid-B560M-H-V2/checkpoints/model.pth"
checkpoint = torch.load(PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

online_network = ResNet18(**config['network']).to(device)
online_network.load_state_dict(state_dict=checkpoint["online_network_state_dict"])

predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
# predictor.load_state_dict(checkpoint["predictor_state_dict"])

target_network = ResNet18(**config['network']).to(device)
target_network.load_state_dict(state_dict=checkpoint["target_network_state_dict"])

# x = torch.randn(1,3,256,256).to(device)
# y = torch.randn(1,3,256,256).to(device)

IMG_PATH1 = "/home/zoid/Documents/Eresh/StrongSORT-YOLO/crop_pad/5.jpg"
IMG_PATH2 = "/home/zoid/Documents/Eresh/StrongSORT-YOLO/crop_pad/57.jpg"

x = torch.unsqueeze(image_to_tensor(IMG_PATH1).to(device),0)
y = torch.unsqueeze(image_to_tensor(IMG_PATH2).to(device),0)
predictions_from_view_1 = predictor(online_network(x))
targets_to_view_1 = target_network(y)

print(predictions_from_view_1,targets_to_view_1)

print(regression_loss(predictions_from_view_1 , targets_to_view_1))
print(targets_to_view_1.shape , predictions_from_view_1.shape)
