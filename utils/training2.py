from utils.common import AverageMeter
import numpy as np
import time

import torch

from utils.common import quaternion_angular_error
from utils.dataloaders import get_dataloaders
from tqdm import tqdm
import os
import copy
import random


def train_model(model, dataloader_list, criterion, optimizer, save_dir = None, num_epochs=25):
    '''
    Args:
        model: The NN to train
        dataloader_list: A dictionary containing loaders (00, 01, ..., 10)
        criterion: The Loss function
        optimizer: Pytroch optimizer. The algorithm to update weights 
        num_epochs: How many epochs to train for
        save_dir: Where to save the best model weights that are found. Using None will not write anything to disk.

    Returns:
        model: The trained NN
        tr_loss_history: list, training Loss history. Recording freq: one epoch.
        val_loss_history: list, validation Loss history. Recording freq: one epoch.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")
    dataloaders = {}
    ## USE 04 for val for now ##
    dataloaders['val'] = get_dataloaders(dset_id="04", batch_size=16)
    ## USE 04 for val for now ##
    dataloaders['train'] = dataloader_list[0]

    dset_num = np.arange(len(dataloader_list))

    val_loss_history = []
    tr_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1000.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        if epoch % 50 == 0 and epoch != 0:
            idx = int (random.choice(dset_num))
            dataloaders['train'] = dataloader_list[idx]
            print("========= Train on Dataset =======", idx)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # loss and number of correct prediction for the current batch
            running_loss = 0.0

            # Iterate over data.
            # TQDM has nice progress bars
            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # For "train" phase, compute the outputs, calculate the loss, update the model parameters
                # For "val" phase, compute the outputs, calculate the loss
                if phase == 'train':
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                # save the best model weights
                # ================================ IMPORTANT ===============================================
                # Lossing connection to colab will lead to loss of trained weights.
                # You can download the trained weights to your local machine. 
                # Later, you can load these weights directly without needing to train the neural networks again.
                # ==========================================================================================
                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir, 'PoseNet.pth'))

            # record the train/val accuracies
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            else:
                tr_loss_history.append(epoch_loss)
                
    print('Best val Loss: {:4f}'.format(best_val_loss))

    return model, tr_loss_history, val_loss_history