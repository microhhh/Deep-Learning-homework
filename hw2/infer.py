import sys

sys.path.append('../')

import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import *

from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
from PIL import Image
# from cnn_visiual import cnn_layer_visualization as visual
from NetC import *

batch_size = 32
workers = 2
num_classes = 65

base_path = '/workspace/fubo/'
train_path = base_path + 'train/'
valid_path = base_path + 'valid/'
test_path = base_path + 'test_list/'

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

model = Net(num_classes)

state_dict = torch.load('netC.pth')
model.load_state_dict(state_dict.state_dict())

print('=> loaded checkpoint')

model = model.to(device)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageData(data.Dataset):
    def __init__(self, root, loader=default_loader, transform=None):
        super(ImageData, self).__init__()
        self.root = root
        self.files = sorted([file for file in os.listdir(self.root) if is_image_file(file)])
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.transform(self.loader(os.path.join(self.root, self.files[index])))

    def __repr__(self):
        return 'ImageData'


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset_train = datasets.ImageFolder(root=train_path, transform=train_transform)
dataset_valid = datasets.ImageFolder(root=valid_path, transform=valid_transform)
dataset_test = ImageData(root=test_path, transform=valid_transform)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

valid_loader = torch.utils.data.DataLoader(
    dataset=dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)


# def extract_feature(loader, model):
#     model.eval()
#     targets = []
#     features = []
#     with torch.no_grad():
#         for i, (data, target) in enumerate(loader):
#             data = data.to(device)
#             targets.append(target.to(device))
#             features.append(model.features(data).view(data.shape[0], -1))
#
#     return torch.cat(targets, 0), torch.cat(features, 0)
#
#
# targets, features = extract_feature(train_loader, model)
# mask = (45 < targets) & (targets < 56)
# tic = time.time()
# perp = 100
# tsne = manifold.TSNE(perplexity=perp, n_iter=2000)
# Y_q = tsne.fit_transform(features[mask].cpu())
# print("perp_{} done, {}s passed".format(perp, time.time() - tic))
#
# plt.scatter(Y_q[:, 0], Y_q[:, 1], s=20, c=targets[mask].cpu(), cmap=plt.get_cmap("tab10"))
# plt.axis('tight')
# plt.savefig('./tsne.pdf')


# def inference(loader, model):
#     model.eval()
#
#     targets = []
#     preds = []
#     pred_probs = []
#     with torch.no_grad():
#         for i, (data, target) in enumerate(loader):
#             data = data.to(device, non_blocking=True)
#             targets.append(target.to(device, non_blocking=True))
#             probs = F.softmax(model(data), dim=1)
#             pred_prob, pred = probs.topk(k=1, dim=1, largest=True, sorted=True)
#             preds.append(pred.reshape(-1))
#             pred_probs.append(pred_prob.reshape(-1))
#
#     return torch.cat(targets, 0), torch.cat(preds, 0), torch.cat(pred_probs, 0)
#
#
# targets, preds, pred_probs = inference(valid_loader, model)
# print((targets == preds).float().mean())
# confidence = pred_probs
# correct = (targets == preds).cpu()
#
# arange = 1 + np.arange(confidence.shape[0])
# xs = arange / confidence.shape[0]
# correct_tmp = correct[confidence.sort(descending=True)[1]]
# accuracies = np.cumsum(correct_tmp.numpy()) / arange
# plt.plot(xs, accuracies)
# plt.grid()
#
# mask = pred_probs < 1.1
# cm = confusion_matrix(targets.cpu()[mask], preds.cpu()[mask])
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.savefig('confusion_matrix_MyModel.pdf')

#
# def conv_visiual(model):
#     feature = model.feature

    # layer_vis = visual.CNNLayerVisualization(feature, 0, 0)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 0, 2)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 0, 4)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 4, 1)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 4, 12)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 4, 15)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 8, 0)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 8, 16)
    # layer_vis.visualise_layer_with_hooks()
    #
    # layer_vis = visual.CNNLayerVisualization(feature, 8, 2)
    # layer_vis.visualise_layer_with_hooks()

    # Layer visualization with pytorch hooks


# conv_visiual(model)

def inference_without_label(loader, model):
    model.eval()

    preds = []
    pred_probs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            probs = F.softmax(model(data), dim=1)
            pred_prob, pred = probs.topk(k=1, dim=1, largest=True, sorted=True)
            preds.append(pred.reshape(-1))

    return torch.cat(preds, 0)


preds = inference_without_label(test_loader, model)

idx_to_class = {i: c for c, i in train_loader.dataset.class_to_idx.items()}
with open('./test_results.txt', 'w') as f:
    f.write('id, label\n')
    for id, pred in zip(test_loader.dataset.files, preds.cpu().numpy()):
        f.write('{}, {} \n'.format(id[:-4], idx_to_class[pred]))

# def inference_without_label(loader, model):
#     model.eval()
#
#     preds = []
#     pred_probs = []
#     with torch.no_grad():
#         for i, data in enumerate(loader):
#             data = data.to(device)
#             probs = F.softmax(model(data), dim=1)
#             pred_prob, pred = probs.topk(k=1, dim=1, largest=True, sorted=True)
#             preds.append(pred.reshape(-1))
#             pred_probs.append(pred_prob.reshape(-1))
#     # print(preds)
#     return torch.cat(preds, 0), torch.cat(pred_probs, 0)
#
#
# preds, pred_probs = inference_without_label(test_loader, model)
#
# idx_to_class = {i: c for c, i in train_loader.dataset.class_to_idx.items()}
# with open('./test_results.txt', 'w') as f:
#     f.write('id, label, probs\n')
#     for id, pred, pred_probs in zip(test_loader.dataset.files, preds.cpu().numpy(), pred_probs.cpu().numpy()):
#         f.write('{}, {},{} \n'.format(id[:-4], idx_to_class[pred], pred_probs))
