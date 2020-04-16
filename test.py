import torch
from datasets.KittiDset import KittiDset
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from utils.dataloaders import get_dataloaders

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def train_val(model, val_set="04"):
    #val
    val_img_path = "/home/hsuan/dataset/sequences/" + val_set + "/image_0"
    val_txt_path = "/home/hsuan/dataset/poses/" + val_set + ".txt"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dset_temp = KittiDset(val_img_path, val_txt_path, transform = transform)

    dataloaders_val = DataLoader(dset_temp, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_net = []
    y_net = []
    z_net = []
    model.eval()
    for inputs, labels in dataloaders_val:
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) # N x 6
            outputs = outputs.cpu().numpy()
            x_net.extend(outputs[:,0])
            y_net.extend(outputs[:,1])
            z_net.extend(outputs[:,2])
    ##################################################
    # ground truth
    filename = val_txt_path
    poses_str = open(filename,'r')
    line_list = []
    for line_str in poses_str.readlines():
        line = np.fromstring(line_str, dtype=float, sep=' ')
        line_list.append(line)
        poses = np.stack(line_list)   
    print(poses.shape)

    x_gt = poses[:,3]
    y_gt = poses[:,7]
    z_gt = poses[:,11]

    ##################################################
    # neural network
    x = x_net
    y = y_net
    z = z_net

    ##################################################
    # visualization
    fig = plt.figure(figsize=(30,20))
    ax = plt.gca(projection='3d',adjustable='box')
    # ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot3D(x_gt, y_gt, z_gt, 'Green') # ground truth
    ax.plot3D(x, y, z, 'Red')            # neural network
    ax.view_init(0, 90)                  # top view
    plt.grid()
    plt.show()
    
    ###  plot separately ###
    plt.plot(x_gt)
    plt.plot(x_net)
    plt.title('X')
    plt.figure()
    plt.plot(y_gt)
    plt.plot(y_net)
    plt.title('Y')
    plt.figure()
    plt.plot(z_gt)
    plt.plot(z_net)
    plt.title('Z')