import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import re

###############
# Dataset class
###############

class dataset(Dataset):
    '''
    Dataset class for chess data from dataset 2: each row is a board state, with 64 features for each square on the chess board, and then 2 features for the move (move from square, move to square)
    NOTE: currently using 1D vector as input; should I reshape to 8x8 and use 2D CNN instead?
    '''

    def __init__(self, set, debug_subset=False): 
        self.dpath = '../Data/CHESS_DATA_kaggle'
        all_data = pd.read_csv(f'{self.dpath}/CHESS_DATA_encoded.csv') # already normalized and 1-hot encoded
        if debug_subset:
            all_data = all_data[int(len(all_data)/8):]
        test_idxs = np.load(f'{self.dpath}/CHESS_DATA_test_idxs.npy')
        if set == 'train':
            use_data = all_data[~all_data['index'].isin(test_idxs)]
        elif set == 'val':
            use_data = all_data[all_data['index'].isin(test_idxs)]
        self.use_x = np.array(use_data.drop(columns=['index', 'MOVE_FROM', 'MOVE_TO']).values)
        self.use_y = np.array(use_data[['MOVE_FROM', 'MOVE_TO']].values)

    def __len__(self):
        return len(self.use_x)

    def __getitem__(self, index):

        # Get input
        x = self.use_x[index]

        # Get labels (one-hot encoded)
        y = self.use_y[index]
        labels = np.zeros((2, 64)) # 64 classes for move_from and 64 classes for move_to
        labels[0, y[0]] = 1 # set the correct move_from class
        labels[1, y[1]] = 1 # set the correct move_to class

        return x, labels
    

def check_inputs(train_ds, train_loader):
    '''
    Check data is loaded correctly
    '''
    print('Train data:')
    print(f'\t{len(train_ds)} obs, broken into {len(train_loader)} batches')
    train_input, train_labels = next(iter(train_loader))
    in_shape = train_input.size()
    print(f'\tEach batch has data of shape {train_input.size()}, and labels of shape {train_labels.size()}')#  (should be [batch_size, 64] for data and [batch_size, 2, 64] for labels)')

    
######################
# For dataset creation
######################

# Modified from "tocsv" by https://github.com/vitiral/fencsv/blob/master/fencsv.py
def fen_to_state(fenstr):
    
    # Set possible pieces and spaces
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))

    # Parse the pieces and add to board
    fen = fenstr.split()
    boardstr, desc = fen[0], fen[1:]
    board = [] # [desc]  # desc is the header # REMOVE THIS SO THAT FIRST LINE IS NOT THE NON-POSITIONAL STATE
    cur_row = []
    for i, c in enumerate(boardstr):
        if c in pieces:
            cur_row.append(c)
        elif c == '/':  # new row
            board.append(cur_row)
            cur_row = []
        elif int(c) in valid_spaces:
            cur_row.extend(' ' * int(c))
        else:
            raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
    board.append(cur_row) 
    board = np.array(board)

    # Check which color is "active", and if its not the one that is currently on top, flip the board (my dataloader assumes truth move is for the player on top)
    first_piece = re.sub(r'\d+', '', fenstr).replace('/','')[0]
    first_color = 'b' if first_piece.islower() else 'w'
    active_color = fenstr[fenstr.find(' ')+1]
    #print(fenstr); print('First color:', first_color); print('Active color: ', active_color)
    if active_color != first_color:
        #print(board)
        board = np.flip(board, axis=0)
        #print(board); print('')
       
    # Return as flattened list
    board = board.flatten().tolist()

    return board


def uci_to_move(uci_string):
    
    positions_uci = [let+str(num) for let in ['a','b','c','d','e','f','g','h'] for num in [1,2,3,4,5,6,7,8]]
    pos_map = dict(zip(positions_uci, np.linspace(1,64,64, dtype=int)))
    move_from = pos_map[uci_string[:2]]
    move_to = pos_map[uci_string[2:4]]
    
    return move_from, move_to