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
MAX_AGE = 30
counter = 0
cropsPath = "/home/anunay/Documents/anunay/savedCrops"


finalPath = "/home/anunay/Documents/anunay/finalwithID"
idCounter = 1

cropsPathsList = os.listdir(cropsPath)
cropsPathsList = natsorted(cropsPathsList, key=lambda y: y.lower())
archivingList = []
print(cropsPathsList)
for i in cropsPathsList:
    # try:
    if os.path.exists(cropsPath + "/" + i):
        img1_path = cropsPath + "/" + i
        img1 = image_to_tensor(img1_path).to(device)
        frameNumber = i.split("f")[-1][:-4]
        print(frameNumber)
        for j in cropsPathsList:
            try:
                if int(j.split("f")[-1][:-4]) <= int(frameNumber) + MAX_AGE:
                    img2_path = cropsPath + "/" + j
                    img2 = image_to_tensor(img2_path).to(device)
                    batch_view_1 = torch.stack([img1,img2])
                    batch_view_2 = torch.stack([img2,img1])

                    predictions_from_view_1 = predictor(online_network(batch_view_1))
                    predictions_from_view_2 = predictor(online_network(batch_view_2))

                    targets_to_view_2 = target_network(batch_view_2)
                    targets_to_view_1 = target_network(batch_view_1)

                    loss1 = regression_loss(predictions_from_view_1 , targets_to_view_1)
                    loss2 = regression_loss(predictions_from_view_2 , targets_to_view_2)

                    print("loss1 = " , loss1)
                    print("loss2 = " , loss2)

                    loss = (loss1 + loss2)/2
                    loss = loss.detach().cpu().numpy().mean()
                    print(loss)
                    if loss <= similarityThreshold:
                        print("Similar object is found")
                        os.rename(img2_path, finalPath + "/" + str(idCounter) + "_"  + j.split("_")[-1])
                    else:
                        print("############################################################################")
                        print("############################################################################")
                        print("############################################################################")
                        pass
            except Exception as e:
                print("EXCEPTION IS: ", e)
        # if os.path.exists(img1_path) == True:
        # os.rename(img1_path, finalPath + "/" + str(idCounter) + "_" + i.split("_")[-1])
    idCounter += 1
        # except Exception as e:
        #      print("KUTCH NAHI HUA")
        #      pass






