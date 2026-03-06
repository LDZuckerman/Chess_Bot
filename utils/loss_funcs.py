import torch
import torch.nn.functional as F
import numpy as np
try:
    import  loss_funcs 
except ModuleNotFoundError:
    from utils import loss_funcs 

def Seperated_CE(preds, targs):

    # # Set predictions that are invalid moves to [???] so that they get maxmium loss
    # invalid_pred_idxs = run_utils.detect_invalid(predictions) 
    # predictions[:, :, invalid_pred_idxs]

    # Seperate from-square and to-sqaure preds and targets
    from_preds = preds[:, 0, :] # CHANGE BECAUSE CHANGED MODEL TO BE CONSISTENT WITH OTHER MODEL OUTPUT SHAPE # from_preds = predictions[:, :64]
    to_preds = preds[:, 1, :] # CHANGE BECAUSE CHANGED MODEL TO BE CONSISTENT WITH OTHER MODEL OUTPUT SHAPE # to_preds = predictions[:, 64:]
    from_targets = targs[:, 0, :] # ADD, TO AVOID HAVING TO SEPERATE BEFOREHAND AND PASS IN from_targets AND to_targets SEPERATELY
    to_targets = targs[:, 1,:] # ADD, TO AVOID HAVING TO SEPERATE BEFOREHAND AND PASS IN from_targets AND to_targets SEPERATELY

    # Compute loss
    loss_from = F.cross_entropy(from_preds, from_targets)
    loss_to = F.cross_entropy(to_preds, to_targets)

    return loss_from + loss_to