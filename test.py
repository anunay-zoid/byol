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

PATH = "/home/anunay/Documents/byol/runs/Sep30_22-49-35_anunay-Legion-5-Pro-16ACH6H/checkpoints/bitches/model90.pth"
checkpoint = torch.load(PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

online_network = ResNet18(**config['network']).to(device)
online_network.load_state_dict(state_dict=checkpoint["online_network_state_dict"])

predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
predictor.load_state_dict(checkpoint["predictor_state_dict"])

target_network = ResNet18(**config['network']).to(device)
target_network.load_state_dict(state_dict=checkpoint["target_network_state_dict"])

# x = torch.randn(1,3,256,256).to(device)
# y = torch.randn(1,3,256,256).to(device)

IMG_PATH1 = "/home/anunay/Documents/crop_pad/5.jpg"
IMG_PATH2 = "/home/anunay/Documents/crop_pad/660.jpg"
IMG_PATH3 = "/home/anunay/Documents/crop_pad/6.jpg"

x = image_to_tensor(IMG_PATH1).to(device)
y = image_to_tensor(IMG_PATH2).to(device)
z = image_to_tensor(IMG_PATH3).to(device)
# predictions_from_view_1 = predictor(online_network(torch.stack([x,z])))
# targets_to_view_1 = target_network(torch.stack([z,x]))

batch_view_1 = torch.stack([x,y])
batch_view_2 = torch.stack([y,x])

predictions_from_view_1 = predictor(online_network(batch_view_1))
predictions_from_view_2 = predictor(online_network(batch_view_2))

targets_to_view_2 = target_network(batch_view_2)
targets_to_view_1 = target_network(batch_view_1)

loss1 = regression_loss(predictions_from_view_1 , targets_to_view_1)
loss2 = regression_loss(predictions_from_view_2 , targets_to_view_2)

print("loss1 = " , loss1)
print("loss2 = " , loss2)

loss = (loss1 + loss2)/2
print("final loss = " , loss)
