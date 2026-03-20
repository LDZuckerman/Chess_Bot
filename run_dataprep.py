import gzip
import os, shutil
import pandas as pd
import numpy as np
import pickle
import json
import sys
import importlib
sys.path.append('Chess_Bot')
from utils import data_utils
import argparse 

def data_prep(): 
    
    # Set stop value for data read-in - get memory issues reading in the whole thing
    stop = 80000 # this should be more than enough
    count = 0
    
    # Read in data
    data = []
    with open('../Data/lichess_db_eval.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
            if count == stop:
                break
            count += 1       
    print(len(data))

    # Save to df
    data_list = []
    for i in range(len(data)):
        fen = data[i]['fen']
        state = data_utils.fen_to_state(fen) # list of length 64
        uci = data[i]['evals'][0]['pvs'][0]['line'].split(' ')[0]
        move_from, move_to = data_utils.uci_to_move(uci)
        dat = state + [move_from, move_to]
        data_list.append(dat) # list of length 66 (64 for board, 2 for move_to and move_from)
    data = pd.DataFrame(data=data_list, columns=[let+str(num) for let in ['a','b','c','d','e','f','g','h'] for num in [1,2,3,4,5,6,7,8]]+['MOVE_FROM', 'MOVE_TO'])

    # Encode
    x_map = {'p':-6, 'n':-5, 'b':-4, 'r':-3, 'q':-2, 'k':-1, ' ':0, 'P':1, 'N':2, 'B':3, 'R':4, 'Q':5, 'K':6} # putting black on top
    data_out = data.copy()
    for col in data.columns[:-2]:
        data_out[col] = data[col].map(x_map)
    data_out.reset_index(inplace=True) # save with "index" column for later use
    data_out.to_csv("../Data/CHESS_DATA_lichess/CHESS_DATA_encoded.csv", index=False)

    # Save test idxs
    all_idxs = data_out['index'].values
    n_test = int(len(all_idxs)*0.2)
    test_idxs = np.random.choice(all_idxs, n_test, replace=False)
    np.save("../Data/CHESS_DATA_lichess/CHESS_DATA_test_idxs.npy", test_idxs)
    

if __name__ == "__main__":

    data_prep()