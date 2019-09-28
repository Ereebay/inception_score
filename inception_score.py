import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import os
from dataset import *
from torchvision.models.inception import inception_v3
from model import *

import numpy as np
from scipy.stats import entropy
ROOT_DIR = '/home/eree/exp'



# def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
#     N = len(imgs)
#
#     assert batch_size > 0
#     assert N > batch_size
#
#     # Set up dataloader
#     dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
#
#     # Load inception model
#     inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
#     inception_model.eval();
#     up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
#     def get_pred(x):
#         if resize:
#             x = up(x)
#         x = inception_model(x)
#         return F.softmax(x).data.cpu().numpy()
#
#     # Get predictions
#     preds = np.zeros((N, 1000))
#
#     for i, batch in enumerate(dataloader, 0):
#         batch = batch.type(dtype)
#         batchv = Variable(batch)
#         batch_size_i = batch.size()[0]
#
#         preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
#
#     # Now compute the mean kl-div
#     split_scores = []
#
#     for k in range(splits):
#         part = preds[k * (N // splits): (k+1) * (N // splits), :]
#         py = np.mean(part, axis=0)
#         scores = []
#         for i in range(part.shape[0]):
#             pyx = part[i, :]
#             scores.append(entropy(pyx, py))
#         split_scores.append(np.exp(np.mean(scores)))
#
#     return np.mean(split_scores), np.std(split_scores)

def compute_inception_score(predictions, num_splits=1):
    print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

if __name__ == '__main__':

    dataset = {x: ImageFolder(root=os.path.join(ROOT_DIR, x)) for x in ['model1', 'model2', 'model3','model1-128']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(dataset[x], batch_size=32, shuffle=True, drop_last=False,
                                                       num_workers=4) for x in ['model1', 'model2', 'model3','model1-128']}

    inception_model = INCEPTION_V3()
    inception_model = inception_model.cuda()
    inception_model.eval()
    for model in ['model1', 'model2', 'model3','model1-128']:
        predictions = []
        for step, data in enumerate(dataloaders_dict[model]):
            data = data.cuda()
            pred = inception_model(data)
            predictions.append(pred.data.cpu().numpy())
            # print(len(predictions))
        predictions = np.concatenate(predictions, 0)
        mean, std = compute_inception_score(predictions, 10)
        predictions = []
        print(f'{model}:mean:{mean},std:{std} ')

