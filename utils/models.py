import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np


############
# 1D CNN
# NOTE: currently using 1D vector as input; should I reshape to 8x8 and use 2D CNN instead? 
############

class Linear_1D(nn.Module):
    '''
    *Very* simple NN of linear layers to work with data from dataset 2: [64] -> [2, 64]
        - 64 input features (each square on the chess board)
        - 2 output features (move from square, move to square) with 64 classes each (which square to move from/to)
    '''
    def __init__(self, in_channels=64, hidden_channels=[32, 16, 32], num_classes=64):
        super(Linear_1D, self).__init__()

        self.activation = nn.ReLU()

        # self.linears = nn.ModuleList()
        # self.linears.append(nn.Linear(in_channels, hidden_channels[0]))
        # for i in range(len(hidden_channels)-1):
        #     self.linears.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
        #     self.linears.append(nn.BatchNorm1d(hidden_channels[i+1]))
        # self.linears.append(nn.Linear(hidden_channels[-1], num_classes))


        self.trunk = nn.ModuleList()
        self.trunk.append(nn.Linear(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels)-1):
            self.trunk.append(nn.Linear(hidden_channels[i], hidden_channels[i+1]))
            self.trunk.append(nn.BatchNorm1d(hidden_channels[i+1]))

        self.final_1 = nn.Linear(hidden_channels[-1], num_classes)
        self.final_2 = nn.Linear(hidden_channels[-1], num_classes)

        
    def forward(self, x):

        # Compute trunk
        x = x.float()
        for layer in self.trunk:
            x = self.activation(layer(x))

        # Probabilities of each class for move_from
        out_1 = self.final_1(x)
        out_1 = nn.Softmax(dim=1)(out_1) # apply softmax to get constrain probabilities to sum to 1
                
        # Probabilities of each class for move_to
        out_2 = self.final_2(x)
        out_2 = nn.Softmax(dim=1)(out_2) # apply softmax to get constrain probabilities to sum to 1

        # Concatenate outputs for move_from and move_to
        out = torch.stack([out_1, out_2], dim=1)

        return out
    

##########################################
# SUGGESTED - DO NOT USE WITHOUT MODIFYING
#   -> Why not just do np.reshape(input, (8, 8)) and then normal CNN?
###########################################

class CNN_TEST(nn.Module):
    def __init__(self, num_piece_types=13, embedding_dim=16):
        """s
        num_piece_types:
            e.g.
            0 = empty
            1-6 = white pawn, knight, bishop, rook, queen, king
            7-12 = black pawn, knight, bishop, rook, queen, king
        """
        super(CNN_TEST, self).__init__()

        # Learn embedding for each piece type
        self.embedding = nn.Embedding(num_piece_types, embedding_dim)

        # Convolutional layers to capture board structure
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Two heads: from-square and to-square
        self.from_head = nn.Linear(256, 64)
        self.to_head = nn.Linear(256, 64)

    def forward(self, x):
        """
        x shape: (batch_size, 64)
        """

        # Embed each square
        # print(x.shape, min(x.flatten()), max(x.flatten()))
        x = x + 6 # because embedding expects 0 - 13
        #print(x.shape, min(x.flatten()), max(x.flatten()))
        x = self.embedding(x.long())  # (batch_size, 64, embedding_dim)

        # Reshape to board format
        x = x.view(-1, 8, 8, x.shape[-1])  # (batch, 8, 8, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (batch, embed_dim, 8, 8)

        # Convolutional feature extraction
        x = self.conv_layers(x)

        # Flatten
        #print(x.shape)
        x = x.reshape(x.size(0), -1) # x.view(x.size(0), -1)

        # Fully connected processing
        x = self.fc(x)

        # Output heads
        from_logits = self.from_head(x)  # (batch, 64)
        to_logits = self.to_head(x)      # (batch, 64)

        # Concatenate outputs (as requested)
        output = torch.stack([from_logits, to_logits], dim=1) # CHANGE FOR CONSISTENCY IN OUTPUT SHAPE WITH OTHER MODELS, WAS torch.cat([from_logits, to_logits], dim=1), OUTPUTING [batch, 128]

        return output

