"""
Created on Mon Mar 22 10:43:25 2021

@author: xuegeeker
@blog: https://github.com/xuegeeker
@email: xuegeeker@163.com
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from PIL import Image 
import numpy as np
import h5py
from operator import truediv
import matplotlib.pyplot as plt


def get_train_test_data(file_name='./Trento_16_16_63_32_32_3.h5', percent = 0.05):
    
    f=h5py.File(file_name, 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    indices = np.arange(data.shape[0])
    shuffled_indices = np.random.permutation(indices)
    images = data[shuffled_indices]
    labels = label[shuffled_indices]
    
    y = np.array([np.arange(6)[l==1][0] for l in labels])
    n_classes = y.max()+1
    i_labeled = []
    
    for c in range(n_classes):

        if (c == 0):
            num_perclass = 129
        elif (c == 1):
            num_perclass = 125
        elif (c == 2):
            num_perclass = 105
        elif (c == 3):
            num_perclass = 154
        elif (c == 4):
            num_perclass = 184
        elif (c == 5):
            num_perclass = 122
        
        i = indices[y==c][:num_perclass]
        i_labeled += list(i)
    l_images = images[i_labeled]
    l_labels = y[i_labeled]

    return l_images, l_labels, data, np.argmax(label,1)

class hsi_dataset(Dataset):
    def __init__(self, data, label):

        self.data = data[:,:16*16*63].reshape(-1,16,16,63)
        self.data = np.swapaxes(self.data,2,3)
        self.data = np.swapaxes(self.data,1,2)

        self.data1 = data[:,16*16*63:(16*16*63+32*32*3)].reshape(-1,32,32,3)
        self.data1 = np.swapaxes(self.data1,2,3)
        self.data1 = np.swapaxes(self.data1,1,2)
        
        self.data2 = data[:,(16*16*63+32*32*3):].reshape(-1,32,32,3)
        self.data2 = np.swapaxes(self.data2,2,3)
        self.data2 = np.swapaxes(self.data2,1,2)

        self.label = label
        self.classes = label.max()+1

    def __getitem__(self, index):

        img1 = self.data[index,:,:,:]
        img1 = torch.from_numpy(img1)

        img2 = self.data1[index,:,:,:]
        img2 = torch.from_numpy(img2)
        
        img3 = self.data2[index,:,:,:]
        img3 = torch.from_numpy(img3)

        return img1, img2, img3, self.label[index]

    def __len__(self):

        return len(self.data)
    
    
def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
    

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0])/255.
        if item == 2:
            y[index] = np.array([0, 100, 100])/255.
        if item == 3:
            y[index] = np.array([100, 0, 100])/255.
        if item == 4:
            y[index] = np.array([128, 128, 0])/255.
        if item == 5:
            y[index] = np.array([200, 100, 0])/255.
        if item == 6:
            y[index] = np.array([255, 100, 100])/255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0
