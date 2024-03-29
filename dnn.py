import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ResNetMicrobiome(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMicrobiome, self).__init__()
        # Load a pre-trained ResNet
        self.model = models.resnet18(pretrained=True)
        # Replace the last fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class CNNMicrobiome(nn.Module):
    def __init__(self, num_classes):
        super(CNNMicrobiome, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3 = nn.Conv2d(40, 60, kernel_size=5)
        self.fc1 = nn.Linear(60 * 4 * 4, 100)  # Adjust the size
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 60 * 4 * 4)  # Adjust the size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def embed_tree_to_matrix(tree, L, S):
    # L: Total number of layers in the tree
    # S: Number of taxa in the input vector
    # tree: A dict with nodes as keys and their children and abundance as values

    M = np.zeros((L, len(S)))  # Initialize the matrix with zeros

    node_list = [tree['root']]  # Start with the root node
    for j in range(L):
        next_list = []
        k = 0
        for node in node_list:
            M[j, k] = node['abundance']  # Assign abundance to the matrix
            next_list.extend(node['children'])  # Prepare the next list of nodes
            k += 1
        node_list = next_list

    return M

