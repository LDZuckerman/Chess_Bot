import torch.nn.functional as F

def Seperated_CE(predictions, targets):

    from_preds = predictions[:, 0, :] # CHANGE BECAUSE CHANGED MODEL TO BE CONSISTENT WITH OTHER MODEL OUTPUT SHAPE # from_preds = predictions[:, :64]
    to_preds = predictions[:, 1, :] # CHANGE BECAUSE CHANGED MODEL TO BE CONSISTENT WITH OTHER MODEL OUTPUT SHAPE # to_preds = predictions[:, 64:]

    from_targets = targets[:, 0, :] # ADD, TO AVOID HAVING TO SEPERATE BEFOREHAND AND PASS IN from_targets AND to_targets SEPERATELY
    to_targets = targets[:, 1, :] # ADD, TO AVOID HAVING TO SEPERATE BEFOREHAND AND PASS IN from_targets AND to_targets SEPERATELY

    loss_from = F.cross_entropy(from_preds, from_targets)
    loss_to = F.cross_entropy(to_preds, to_targets)

    return loss_from + loss_to