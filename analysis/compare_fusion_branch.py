'''
分析对比  数字和英语字母表示数字之间的表征差异，
针对例如 0.12 & 0.13, 这两种方法哪种对这两个数字的表征差异更大

'''

import sys

sys.path.insert(0,'./')


import numpy as np
import pandas as pd
import sys
import os
from random import shuffle

from tqdm import tqdm
import torch
import torch.nn as nn


import datetime
from utils import *

# from models.model_graphdrp import GraphDRP
from models.model_transedrp_reg_num import TransEDRP
from models.model_graphdrp_reg_num2 import GraphDRP
# from models.model_graphdrp_reg_KGE import GraphDRP
import argparse
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import torch.nn as nn
import torch
from tqdm import tqdm
from random import shuffle
import pandas as pd
import numpy as np
import clip
from copy import deepcopy

import datetime
import argparse
import random

from utils import copyfile

# training function at each epoch
CUDA_LAUNCH_BLOCKING=1
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
from torch_geometric.data import Data, Batch
            
            
def tsne(X):
    '''t-SNE 理想的表征'''
    length = X.shape[0]
    # X= torch.Tensor(np.array([[i,float(i/1000)] for i in range(0,1000,1)]))
    tsne = TSNE(n_components = 1,  init='pca',n_iter=2500, random_state = 42)
    Y = tsne.fit_transform(X.cpu())
    # print(tsne.embedding_)
    fig = plt.figure(figsize=(8, 8))
    # _, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
    plt.scatter(Y[:, 0], Y[:, 0], c= [i for i in range(0, length, 1)], cmap=plt.cm.Spectral, label='number scale, red is 0, prope is 1000')
    # pdb.set_trace()
    plt.title("num2vec")
    plt.legend()
    plt.savefig("/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/analysis/1000_color_text_num1D.jpg")
    # plt.axis('tight')
    # pdb.set_trace()
    # plt.show()
    
    
    # X= torch.Tensor(np.array([[i,float(i/1000)] for i in range(0,1000,1)]))
    tsne = TSNE(n_components = 2,  init='pca',n_iter=2500, random_state = 42)
    Y = tsne.fit_transform(X.cpu())
    # print(tsne.embedding_)
    fig = plt.figure(figsize=(8, 8))
    # _, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
    plt.scatter(Y[:, 0], Y[:, 1], c= [i for i in range(0, length, 1)], cmap=plt.cm.Spectral, label='number scale, red is 0, prope is 1000')
    # pdb.set_trace()
    plt.title("num2vec")
    plt.legend()
    plt.savefig("/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/analysis/1000_color_text_num.jpg")
    
    return 'ok'

# def tsne(X):
#     '''t-SNE 正常的版本'''

#     # X= torch.Tensor(np.array([[i,float(i/1000)] for i in range(0,1000,1)]))
#     tsne = TSNE(n_components = 1,  init='pca',n_iter=250, random_state = 42)
#     Y = tsne.fit_transform(X.cpu())
#     # print(tsne.embedding_)
#     fig = plt.figure(figsize=(8, 8))
#     # _, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
#     plt.scatter(Y[:, 0], Y[:, 0], c= [i for i in range(0, 1000, 1)], cmap=plt.cm.Spectral, label='number scale, red is 0, prope is 1000')
#     # pdb.set_trace()
#     plt.title("num2vec")
#     plt.legend()
#     plt.savefig("/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/analysis/1000_color_text_num1D.jpg")
#     # plt.axis('tight')
#     # pdb.set_trace()
#     # plt.show()
    
    
#     # X= torch.Tensor(np.array([[i,float(i/1000)] for i in range(0,1000,1)]))
#     tsne = TSNE(n_components = 2,  init='pca',n_iter=250, random_state = 42)
#     Y = tsne.fit_transform(X.cpu())
#     # print(tsne.embedding_)
#     fig = plt.figure(figsize=(8, 8))
#     # _, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
#     plt.scatter(Y[:, 0], Y[:, 1], c= [i for i in range(0, 1000, 1)], cmap=plt.cm.Spectral, label='number scale, red is 0, prope is 1000')
#     # pdb.set_trace()
#     plt.title("num2vec")
#     plt.legend()
#     plt.savefig("/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/analysis/1000_color_text_num.jpg")
    
#     return 'ok'


# def main(model, config, yaml_path, show_type):

#     device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
    
#     # def generate_samples(model, data, start, end, index):
#     # data: list, consist of [drug smile, cell line, ic50]
#     descriptions = []
    
    
#     start,end = 0,1000
#     for ic50 in range(start,end,1):
#         des = "zero point " + num2english(ic50/1000)
#         descriptions.append(des)
        
#     with torch.no_grad():
#         model.eval()
#         text = clip.tokenize(descriptions,context_length=300).to(device)
#         text_features = model.encode_num(text)

#     pdb.set_trace()
#     tsne(text_features)


def predicting(model, device, loader, loader_type, args):
    model.eval()

    data_value_mp = {}
    
    _PATH_data_value_mp = True
    
    if _PATH_data_value_mp:
        print("Load dict pth...")
        data_value_mp = torch.load("train_dict.pth")
    else:
        print("Make prediction for {} samples...".format(len(loader.dataset)))
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(loader)):
                data = data.to(device)
                
                
                for index_item in range(len(data.y)):
                    value = str(data.y[index_item].item()).split('.')[1][:3]
                    if data_value_mp.get(value):
                        data_value_mp[value].append(data[index_item])
                    else:
                        data_value_mp[value] = []
                        data_value_mp[value].append(data[index_item])
        torch.save(data_value_mp, "train_dict.pth")
        
    print("data_value_mp is ok.")
    total_preds = {}
    with torch.no_grad():
        for key in tqdm(data_value_mp):
            temp_preds = torch.Tensor()

            batch = Batch.from_data_list(data_value_mp[key])
            # pdb.set_trace()
            # data_value_mp
          
            batch = batch.to(device)
            x_drug = model.drug_module(batch)
            x_cell = model.cell_module(batch.target[:, None, :])
            _, fusion_features = model.fusion_module(x_drug, x_cell)
                
            # temp = torch.cat((total_preds, torch.mean(temp_preds,0).reshape(1,-1)), 0)
            total_preds[key] = torch.mean(fusion_features, 0)
        # pdb.set_trace()
    print("predict data_value_mp is ok.")
    
    
    return total_preds
    
    
def main(model, config, yaml_path, show_type):

    
    model = model.to(device)
    model.eval()
    
    # def generate_samples(model, data, start, end, index):
    # data: list, consist of [drug smile, cell line, ic50]
    descriptions = []
    
    # pdb.set_trace()
    

    trainval_dataset, test_dataset = load(config)
    
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,\
        [
            int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val']))),\
            len(trainval_dataset)-int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val'])))
        ])
        
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size']['train'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size']['val'], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)


    features_mp = predicting(model, device, train_loader, "test", config)
    # torch.save( features_mp, 'features_mp_train_graphdrp_CLIP_KGE.pth')
    torch.save( features_mp, 'features_mp_train_graphdrp_MSE.pth')
    # features_mp = torch.load("features_mp.pth")
    features_mp_sorted_key = sorted(features_mp)
    # pdb.set_trace()
    features = torch.Tensor()
    # # 排序
    # for i in range(1000):
    #     num = i/1000
    #     num = str(round(num,3)).split('.')[1]
        
    #     while len(num)!=3:
    #         num = num + '0'
        
    #     print(num)
    #     if features_mp.get(num):
            
    #         # pdb.set_trace()
            
    #         features = torch.cat((features, features_mp[num].detach().cpu().reshape(1,-1)), 0)

    for key in features_mp_sorted_key:
        # pdb.set_trace()
        features = torch.cat((features, features_mp[key].detach().cpu().reshape(1,-1)), 0)
    
    
    tsne(features)
  
    
    
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def getConfig():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./config/Transformer_edge_concat_GDSCv2.yaml",
        help="",
        
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="/home/lk/project/NMI_DRP/exp/TransEDRP_NCI60_m2r_30_visual_20230214110510/TransE.model",
        help="",
    )
    
    args = parser.parse_args()
    import yaml

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config, args.config, args.model


if __name__ == "__main__":
    config, yaml_path, infer_model = getConfig()
    cuda_name = config["cuda_name"]
    seed_torch(config["seed"])

    modeling = [GraphDRP, TransEDRP][
        config["model_type"]
    ]
    model = modeling(config)
    
    model.load_state_dict(torch.load(infer_model), strict=False)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    # main(model, config, yaml_path, show_type= 'total')
    main(model, config, yaml_path, show_type= 'byclass')
