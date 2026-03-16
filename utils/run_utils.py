import numpy as np
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import json
#from Chessnut import Game
try:
    import  models, loss_funcs #, plot_utils
except ModuleNotFoundError:
    from utils import models, loss_funcs #, plot_utils


def train_net(loader, model, loss_name, optimizer, device, save_examples=False, save_dir=None, epoch=None):
    '''
    Train superised for one epoch
    '''

    # Train on each batch in train loader
    k = 0
    for data, targets in loader: #batch_idx, (data, targets) in enumerate(loop):
        #print(f'  Batch {k}', end='\r', flush=True); k += 1

        # Set data to be on correct device
        data = data.to(device)
        targets = targets.float().to(device)

        # Forward
        predictions = model(data)
        # print(targets)
        # print(np.argmax(targets.cpu().detach().numpy(), axis=2)) # get true class for move_from and move_to
        # print(predictions)
        # print(np.argmax(predictions.cpu().detach().numpy(), axis=2)) # get class with highest probability for move_from and move_to
        # a=b

        # Loss
        if torch.isnan(predictions).any() or not torch.isfinite(predictions).all():
            raise ValueError('predictions become NaN or inf')
        if loss_name == 'MSE':
            loss_func = nn.MSELoss() #getattr(loss_funcs, loss_name) # e.g. nn.MSELoss()
            loss = loss_func(predictions, targets)
        elif loss_name == 'CE':
            loss_func = nn.CrossEntropyLoss() 
            loss = loss_func(predictions.float(), targets) # loss_func(predictions.view(-1, 64), torch.argmax(targets, dim=2).view(-1))
        elif loss_name == 'Seperated_CE':
            loss_func = getattr(loss_funcs, 'Seperated_CE')
            loss = loss_func(predictions, targets)
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad() 

    # Plot examples from last batch
    # if save_examples:
    #     plot_utils.plot_epoch_examples(data, torch.tensor(targets), torch.tensor(predictions), save_dir, epoch) # data, targets will be those loaded from last batch

    return loss 


def save_model_results(val_loader, save_dir, model):
    '''
    Run each validation obs through model, save results
    '''
    print(f'Loading model back in, saving results on validation data in {save_dir}')
    if os.path.exists(save_dir) == False: os.mkdir(save_dir)
    all_val_preds = []
    all_val_trues = []
    i = 0
    for X, y in val_loader:
        X = X.to('cpu').detach()
        y = y.to('cpu').detach().numpy()
        # if torch.is_tensor(y):
        #     y = y.cpu().detach().numpy()
        out = model(X).cpu().detach().numpy()
        # if out.shape == : # IF THE MODEL IS CREATING PREDS AS ONE VECT OF LEN 128, E.G. FROM CNN_TEST
        #     out = np.reshape(out, ())
        #     print() # 
        #     a=b
        preds = np.argmax(out, axis=2) # get class with highest probability for move_from and move_to
        targs = np.argmax(y, axis=2) # collapse back
        for i in range(len(preds)):
            all_val_preds.append(preds[i])
            all_val_trues.append(targs[i]) 
    
    np.save(f'{save_dir}/all_val_preds.npy', all_val_preds)
    np.save(f'{save_dir}/all_val_trues.npy', all_val_trues)



def get_modelDF(modeldir='', tag='', metrics=['CE']):
    '''
    Helper function to create dataframe of all run models, their parameters, and results 
    '''
    expdirs = [f for f in os.listdir(modeldir) if os.path.isdir(f'{modeldir}/{f}') and tag in f]
    
    # Create DF from exp dicts
    all_info = []
    for expdir in expdirs:
        
        # Skip if not finished training 
        if not os.path.exists(f'{modeldir}/{expdir}/test_preds'):
            print(f'Skipping {expdir}; not finished training')
            continue
        if not os.path.exists(f'{modeldir}/{expdir}/exp_file.json'):
            print(f'Skipping {expdir}; no exp_file found')
            continue
        exp_dict = json.load(open(f'{modeldir}/{expdir}/exp_file.json','rb')) 

        # Set hidden_channels if missing
        if exp_dict['model_name'] == 'Linear_1D': exp_dict['hidden_channels'] = "[32, 16, 32]" if 'hidden_channels' not in exp_dict else exp_dict['hidden_channels']
        elif exp_dict['model_name'] == 'CNN_TEST': exp_dict['hidden_channels'] = "[64, 128, 128]"  if 'hidden_channels' not in exp_dict else exp_dict['hidden_channels']
        
        # Add val mse
        for metric in metrics:
            val = prediction_validation_results(output_dir=f'{modeldir}/{expdir}/test_preds', metric=metric)
            exp_dict[metric] = val
    
        # Add to dict
        all_info.append(exp_dict)
    
    # # Drop cols and sort 
    # try:
    #     all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'true_dir', 'randomSharp','sub_dir','inject_brightpoints'])
    # except KeyError:
    #     all_info = pd.DataFrame(all_info).drop(columns=['img_dir', 'seg_dir', 'randomSharp','sub_dir','inject_brightpoints'])
    # all_info = all_info.sort_values(by='net_name')
    all_info = pd.DataFrame(all_info)

    # # Change col names for compact display
    # all_info = all_info.rename(columns={"learning_rate":"lr","num_epochs":"ne"}) 
    
    return all_info 


def display_DF(DF, ignore_cols):

    DF = DF.drop(columns=ignore_cols)
    display(DF)


def prediction_validation_results(output_dir, metric):
    '''
    Compute average error on validation predictions 
    '''

    
    preds = np.load(f'{output_dir}/all_val_preds.npy')#; print(f'{output_dir}/all_val_preds.npy', type(preds), preds.dtype)
    trues = np.load(f'{output_dir}/all_val_trues.npy', allow_pickle=True)

    if metric == 'RMSE':
        out = np.sqrt(np.nanmean((trues-preds)**2))

    elif metric == 'CE':
        loss = nn.CrossEntropyLoss() 
        try:
            out = loss(torch.tensor(preds).float(), torch.tensor(trues).float()).item()
        except TypeError:
            out = 'Err'
        
    return out




'''
Functions for determining move validity
'''

def detect_invalid(inputs, preds):

    #  Go from class probs to predicted class
    preds = torch.from_numpy(np.argmax(preds.cpu().detach().numpy(), axis=2)).float() # get class with highest probability for move_from and move_to
    inputs = torch.from_numpy(np.argmax(inputs.cpu().detach().numpy(), axis=2)).float() # collapse back

    # Seperate from-square and to-sqaure preds and targets
    from_preds = preds[:, 0, :]
    to_preds = preds[:, 1, :] 

    # 
    out = preds.copy()
    for i in range(len(preds.shape[0])):
        board = inputs[i]; print(board.shape) # should be len 64
        move_from = from_preds[i]; print(move_from) 
        move_to =  to_preds[i]


def board_to_fen(board, turn="w", castling="KQkq", ep="-", halfmove=0, fullmove=1):
    fen_rows = []
    
    for rank in range(8):
        row = ""
        empty_count = 0
        
        for file in range(8):
            value = board[rank * 8 + file]
            piece = INT_TO_FEN[value]
            
            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    row += str(empty_count)
                    empty_count = 0
                row += piece
        
        if empty_count > 0:
            row += str(empty_count)
        
        fen_rows.append(row)
    
    fen_board = "/".join(fen_rows)
    return f"{fen_board} {turn} {castling} {ep} {halfmove} {fullmove}"   

def index_to_square(index):

    file = index % 8
    rank = 8 - (index // 8)
    file_letter = chr(ord('a') + file)

    return f"{file_letter}{rank}"

def move_to_uci(start_idx, end_idx):
    return index_to_square(start_idx) + index_to_square(end_idx)

def is_legal_move(board_array, start_idx, end_idx, turn="w"):

    fen = board_to_fen(board_array, turn=turn)
    game = Game(fen)
    move = move_to_uci(start_idx, end_idx)
    
    return move in game.get_moves()