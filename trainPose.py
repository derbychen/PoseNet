import argparse
from datetime import datetime
import numpy as np
import os
import time
import torch
from torch import nn
from torchvision import transforms, models
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from torchviz import make_dot
from utils.training import train, validate, model_results_pred_gt
from datasets.KittiDset import KittiDset
from utils.common import draw_poses
from utils.common import draw_record
from utils.common import imshow
from utils.common import save_checkpoint
# from utils.common import AverageMeter
from utils.common import calc_poses_params, quaternion_angular_error
from utils.common import draw_pred_gt_poses

from models.posenet import PoseNet, PoseNetCriterion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

from utils.dataloaders import get_dataloaders
from utils.training2 import train_model
from test import train_val

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### change epochs ###
    num_epochs = 30

    ### IO
    # Directory to save weights to
    save_dir = "Weight"
    try:
        os.makedirs(save_dir, exist_ok=True)
    except:pass

    # Choose dataset from 00 to 10. ex Choose_dset = ["01", "07", "10"]
    Choose_dset = ["04"]
    dataloader_list = [get_dataloaders(i, batch_size=16) for i in Choose_dset]

    # Create pretrained feature extractor
    # feature_extractor = models.resnet18(pretrained=True)
    feature_extractor = models.resnet34(pretrained=True)
    # feature_extractor = models.resnet50(pretrained=True)

    # Num features for the last layer before pose regressor
    num_features = 2048

    # Create model
    model = PoseNet(feature_extractor, num_features=num_features, pretrained=True)
    model = model.to(device)

    # Criterion
    # criterion = PoseNetCriterion(stereo=stereo, beta=500.0)
    criterion = PoseNetCriterion(stereo=False, learn_beta=True)
    criterion = criterion.to(device)

    # Add all params for optimization
    param_list = [{'params': model.parameters()}]
    if criterion.learn_beta:
        param_list.append({'params': criterion.parameters()})

    # Create optimizer ## change learning rate here!!!!!!!!!
    optimizer = optim.Adam(params=param_list, lr=1e-4, weight_decay=0.0005)

    # Train the model!
    model_trained, tr_his, val_his = train_model(model=model, dataloader_list=dataloader_list, criterion=criterion, optimizer=optimizer,
            save_dir=save_dir, num_epochs=num_epochs)

    print('\n Finish Training ....')

    x = np.arange(num_epochs)
    plt.figure()
    plt.plot(x, tr_his)
    plt.plot(x, val_his)
    plt.legend(['Train Loss', 'Val Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PoseNet')
    plt.show()

    print('\n Plotting ....')
    train_val(model_trained, "04")

    print('\n Done!!!!!')

if __name__ == "__main__":
    main()