import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target, ignore_true_neg_and_reduce=True):
    smooth = 1e-5

    """
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
    """

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.cpu().numpy()
    output_ = (output > 0.5).reshape(output.shape[0],-1)
    target_ = (target > 0.5).reshape(output.shape[0],-1)
    intersection = (output_ & target_).sum(axis=1)
    union = (output_ | target_).sum(axis=1)
    if ignore_true_neg_and_reduce:
        intersection = intersection[union!=0]
        union = union[union!=0]
        iou = ((intersection + smooth) / (union + smooth)).mean()
    else:
        iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_coef(output, target):
    smooth = 1e-7

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    # target = target.view(-1).data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()

    print(output.shape, target.shape)

    print(intersection, output_.sum(), target_.sum())

    return (2. * intersection + smooth) / \
        (output_.sum() + target_.sum() + smooth)
