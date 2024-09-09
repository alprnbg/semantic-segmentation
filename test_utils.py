import time
import cv2
import torch
import numpy as np


def preprocess(input_path, input_size, is_rgb=True):
    img = cv2.imread(input_path)
    img = cv2.resize(img, input_size)
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img-mean)/std
    img = img.astype('float32')
    return img


def normal_inference(input_path, model, input_size, measure_time, dec_th=0.5):
    img = preprocess(input_path, input_size)
    input = torch.from_numpy(img).permute((2,0,1)).unsqueeze(0).float().cuda()
    if measure_time:
        for _ in range(30):
            t1 = time.time()
            output = model(input)
            t2 = time.time()
            print((t2-t1)*1000)
    else:
        output = torch.sigmoid(model(input))
    output_mask = ((output[0,:,:,:]>dec_th).detach().cpu().numpy()).astype("float32")
    return output_mask


def partition_inference(input_path, model, input_size, partition_size, num_classes, dec_th=0.5):
    img = preprocess(input_path, input_size)
    ps = partition_size
    assert input_size[0] % ps == 0 and input_size[1] % ps == 0
    output_mask = np.zeros((num_classes, input_size[1], input_size[0]), dtype="float32")
    for i in range(input_size[1]//ps):
        for j in range(input_size[0]//ps):
            partition_img = img[i*ps:(i+1)*ps, j*ps:(j+1)*ps,:]
            input = torch.from_numpy(partition_img).permute((2,0,1)).unsqueeze(0).\
                        float().cuda()
            output = torch.sigmoid(model(input))
            output_mask[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = \
                    ((output[0,:,:,:]>dec_th).detach().cpu().numpy()).astype("float32")
    return output_mask
