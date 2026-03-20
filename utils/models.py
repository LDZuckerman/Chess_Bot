import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np


############
# 1D CNN
#  - 1D vector as input; could also reshape to 8x8 and use 2D CNN instead 
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
    
####################################################
# Lets try a unet - they seem to work for everything
####################################################
    
class UNet(nn.Module):
    '''
    UNet for [1, N_pix, N_pix] -> [64, 64] 
    '''
    
    def __init__(self, hidden_channels=[64, 128, 256]): # len_set, k_size=3, padding_mode='zeros', hidden_channels=[16, 64])
        
        super(UNet, self).__init__() 
        
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        in_channels = 1
        for n_chans in hidden_channels:
            self.downs.append(DoubleConv(in_channels, n_chans))
            in_channels = n_chans  # after each convolution we set (next) in_channel to (previous) out_channels 
            
        for n_chans in reversed(hidden_channels):
            self.ups.append(nn.ConvTranspose2d(n_chans*2, n_chans, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(n_chans*2, n_chans))
            
        self.bottleneck = DoubleConv(hidden_channels[-1], hidden_channels[-1]*2)
        self.final_conv = nn.Conv2d(hidden_channels[0], 2, kernel_size=1)
    

    def forward(self, x): 
        
        # Reshape to board image and add "channel" dim  
        x_in = x.clone()
        x = x.to(torch.float32); 
        x = x.reshape(x.shape[0], 1, 8, 8);# print('x', x.shape) 

        # Perform downs
        skip_connections = []#; i=0
        for down in self.downs:
            x = down(x)#; print('down'+str(i), x.shape); i+=1
            skip_connections.append(x) 
            x = self.pool(x)
        x = self.bottleneck(x)#; print('bottleneck', x.shape)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1] # reverse 
        
        # Perform ups
        for idx in range(0, len(self.ups), 2): # step of 2 becasue add conv step
            x = self.ups[idx](x); #print(self.ups[idx], x.shape)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # Perform final 
        x = self.final_conv(x)
        
        # Reshape to target dim (batch, 2, 64)
        x = x.reshape(x_in.shape[0], 2, 64)

        return x
    

class DoubleConv(nn.Module):
    '''
    Containor for conv sets (for convenience)
    '''
    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)
    

##########################################
#  Experimental -> but why not just do np.reshape(input, (8, 8)) and then normal CNN?
###########################################

class CNN_TEST(nn.Module):
    def __init__(self, num_piece_types=13, embedding_dim=16, hidden_channels = [64, 128, 128]):
        """
        num_piece_types:
            e.g.
            0 = empty
            1-6 = our pawn, knight, bishop, rook, queen, king
            7-12 = opponent pawn, knight, bishop, rook, queen, king
        """
        super(CNN_TEST, self).__init__()

        # Learn embedding for each piece type
        self.embedding = nn.Embedding(num_piece_types, embedding_dim)

        # Convolutional layers to capture board structure
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),)
        self.conv_layers = nn.ModuleList()
        in_chans = embedding_dim
        for hc in hidden_channels:
            self.conv_layers.append(nn.Conv2d(in_channels=in_chans, out_channels=hc, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm2d(hc))
            in_chans = hc

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
        #x = self.conv_layers(x)
        for conv in self.conv_layers:
            x = conv(x)

        # Flatten
        #print(x.shape)
        x = x.reshape(x.size(0), -1) # x.view(x.size(0), -1)

        # Fully connected processing
        x = self.fc(x)

        # Output heads
        from_logits = self.from_head(x)  # (batch, 64)
        to_logits = self.to_head(x)      # (batch, 64)

        # Concatenate outputs 
        output = torch.stack([from_logits, to_logits], dim=1) # CHANGE FOR CONSISTENCY IN OUTPUT SHAPE WITH OTHER MODELS, WAS torch.cat([from_logits, to_logits], dim=1), OUTPUTING [batch, 128]

        return output

