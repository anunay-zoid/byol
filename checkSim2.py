import os
import numpy as np
import cv2
import torch
from torchvision import models
import glob
from natsort import natsorted, ns

from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

image_size = 256

def checker(img_name):
     frameNumber = img_name.split("f")[-1][:-4]
     return frameNumber

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

def cosineSimilarity(embedding):
    embedding = embedding.cpu().numpy()
    a = embedding[0]
    b = embedding[1]
    similarity = np.dot(a,b)/(a * b)
    return similarity


PATH = "/home/anunay/Documents/byol/runs/Sep30_22-49-35_anunay-Legion-5-Pro-16ACH6H/checkpoints/bitches/model_80.pth"
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


similarityThreshold = 300
MAX_AGE = 20
counter = 0
cropsPath = "/home/anunay/Documents/anunay/savedCrops"


finalPath = "/home/anunay/Documents/anunay/finalwithID"
idCounter = 1

cropsPathsList = os.listdir(cropsPath)
cropsPathsList = natsorted(cropsPathsList, key=lambda y: y.lower())
archivingList = []
print(cropsPathsList)

for i in range(len(cropsPathsList)):
    if os.path.exists(cropsPath + "/" + cropsPathsList[i]) != True:
        pass
    img = image_to_tensor(cropsPath + "/" + cropsPathsList[i]).to(device)
    img_list = [img]*MAX_AGE
    current_frame = cropsPathsList[i].split("f")[-1][:-4]
    compare_list = []
    indices = []
    j=1
    counter = 0
    while True:
        if checker(cropsPathsList[i+j]) == current_frame:
            pass
        else:
             counter+=1
             compare_list.append(image_to_tensor(cropsPath + "/" + cropsPathsList[i+j]).to(device))
             indices.append(i+j)
             if(counter == MAX_AGE):
                  break

        j+=1
    # for j in range(1,MAX_AGE+1):
    #     img_list = [img]*MAX_AGE
    #     compare_list.append(image_to_tensor(cropsPathsList[i+j]))
    batch_view_1 = torch.stack(img_list)
    batch_view_2 = torch.stack(compare_list)
    print(batch_view_1.shape , batch_view_2)

    predictions_from_view_1 = predictor(online_network(batch_view_1))
    predictions_from_view_2 = predictor(online_network(batch_view_2))

    targets_to_view_2 = target_network(batch_view_2)
    targets_to_view_1 = target_network(batch_view_1)

    loss1 = regression_loss(predictions_from_view_1 , targets_to_view_1)
    loss2 = regression_loss(predictions_from_view_2 , targets_to_view_2)

    print("loss1 = " , loss1)
    print("loss2 = " , loss2)

    loss = (loss1 + loss2)/2
    loss = loss.detach().cpu().numpy()

    for k in range(len(loss)):
        if(loss[k]<=similarityThreshold):
            print("Similar object is found")
            os.rename(cropsPath + "/" + cropsPathsList[indices[k]], finalPath + "/" + str(idCounter) + j.split("_")[-1])
        else:
            print("kuch nahiu mila")
            pass
              

idCounter+=1

