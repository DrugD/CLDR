

import pdb,torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
import random
import os

from tqdm import tqdm


import sys

sys.path.insert(0,"./models")



import math
import pdb
import re
import os
import torch.nn.functional as F
from utils import copyfile
import random
import datetime
from utils import *

from models.model_graphdrp_reg_num import GraphDRP
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

# training function at each epoch
# training function at each epoch
import torch
import os
from utils import num2english
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()



def train(model, device, train_loader, optimizer, epoch, log_interval, args):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    DC_cross_entropy_loss = torch.nn.CrossEntropyLoss()
    T_cross_entropy_loss = torch.nn.CrossEntropyLoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        drug_cell_logits, text_logits = model(data)
        
        # pdb.set_trace()
       
        
        labels = torch.arange(data.y.shape[0]).long().to(device) 
        
        loss_dc = DC_cross_entropy_loss(drug_cell_logits, labels)
        loss_t = T_cross_entropy_loss(text_logits, labels)
        loss = (loss_dc + loss_t)/2
        
        loss.backward()
        
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} ({:.0f}%)\tLoss: {:.6f}".format(
                    epoch, 100.0 * batch_idx / len(train_loader), loss.item()
                )
            )
    return sum(avg_loss) / len(avg_loss)


def generate_samples(model, data):
    # data: list, consist of [drug smile, cell line, ic50]
    descriptions = []
   
    
    sentence_feature = []
    
  
                
    descriptions = []
    
    # for index, item in enumerate(data.y):
    temp = []

    for ic50 in range(0,1000,1):
        des = "zero point " + num2english(ic50/1000)
        temp.append(des)

    descriptions.append(temp)
    text = clip.tokenize(temp,context_length=300).to(device)
    text_features = model.encode_num(text)
     
    return text_features
            
            
def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
            data = data.to(device)
            data_ = deepcopy(data)
            drug_cell_logits, text_logits = model(data)
            
            
            gt_sentence_features = generate_samples(model, data)
            gt_sentence_features=gt_sentence_features/gt_sentence_features.norm(dim=1,keepdim=True)
            # pdb.set_trace()
            # for index_item in range(len(gt_sentence_features)):
            #     gt_sentence_features[index_item] = gt_sentence_features[index_item]/gt_sentence_features[index_item].norm(dim=1, keepdim=True)
            # drug_cell_logits.size()[0] use as batch_size
            gt_sentence_features = gt_sentence_features.repeat(drug_cell_logits.size()[0],1,1)
            
            fusion_features = model.infer(data_)
            fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
            # pdb.set_trace()
            # gt_sentence_features = gt_sentence_features / gt_sentence_features.norm(dim=1, keepdim=True)
            fusion_features = fusion_features.unsqueeze(2)
            
            pred = (model.logit_scale.exp()*torch.bmm(gt_sentence_features,fusion_features)).squeeze()
            
            # pdb.set_trace()
            
            pred = (torch.argmax(pred,1)/1000).cpu().view(-1, 1)
            
            # for index_item in range(len(gt_sentence_features)):
            #     gt_sentence_features[index_item] = gt_sentence_features[index_item]/gt_sentence_features[index_item].norm(dim=1, keepdim=True)
            #     pred = (torch.argmax(model.logit_scale.exp() * fusion_features[index_item] @ gt_sentence_features[index_item].t())/1000).cpu().view(-1, 1)
            total_preds = torch.cat((total_preds, pred), 0)

            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            
            
            # G = total_labels.numpy().flatten()
            # P = total_preds.numpy().flatten()
            # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]
            # print(ret)
            # pdb.set_trace()
            
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


'''freeze'''


def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )


def main(config, yaml_path):


    model = GraphDRP(config)

    model.to(device)

    des = []
  
    for i in tqdm(range(0,1000,1)):
        # INT_number.append(tester.get_embed(i))
        # pdb.set_trace()
        des.append("zero point " + num2english(i/1000))
        
    number = clip.tokenize(des,context_length=300).to(device)
    
    # pdb.set_trace()
    
    INT_number = model.encode_num(number.cuda())


    # Draw


    INT_number = INT_number.cpu().detach()

    embeddings_tSNE = manifold.TSNE(n_components=2, init='pca',n_iter=2500).fit_transform(INT_number)




    start = 0
    stop = 1000
    step = 1


    # embeddings_tSNE = TSNE(n_components=2, init='pca',n_iter=250, random_state = seed).fit_transform(embeddings)
    embeddings_tSNE = pd.DataFrame(embeddings_tSNE, columns=['tSNE_1', 'tSNE_2'])


    embeddings_tSNE['magnitude'] = [i for i in range(start, stop, step)]
    plt.scatter(embeddings_tSNE.tSNE_1, embeddings_tSNE.tSNE_2, c=embeddings_tSNE.magnitude, s=10)
    plt.title('2-D t-SNE Visualization of 1000 Integers’ Embedding Vectors')
    plt.xlabel('tSNE_1')
    plt.ylabel('tSNE_2')

    plt.savefig("/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/exp/GAT_GCN_number/text_num_GDSCv2__20231011195851/iter_2500.jpg")
    plt.close()



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
    args = parser.parse_args()
    import yaml

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config, args.config


def load_weight(pretrain_model, model, remainRaw=False):
    pre_dict = {}

    pretrain_weight = torch.load(
        pretrain_model, map_location=torch.device('cpu'))

    for key, value in pretrain_weight.items():

        if "drug_module" in key or "cell_module" in key:
            pre_dict[key] = value
        else:
            if remainRaw:
                key_names = [key,
                             key.replace("fusion_module", "fusion_module1"),
                             key.replace("fusion_module", "fusion_module2")
                             ]
                pre_dict[key_names[2]] = value
            else:
                key_names = [key.replace("fusion_module", "fusion_module1"),
                             key.replace("fusion_module", "fusion_module2")
                             ]
            pre_dict[key_names[0]] = value
            pre_dict[key_names[1]] = value

    model.load_state_dict(pre_dict, strict=True)

    return model



if __name__ == "__main__":
    config, yaml_path = getConfig()
    seed_torch(config["seed"])

    cuda_name = config["cuda_name"]

    print("CPU/GPU: ", torch.cuda.is_available())

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    main(config, yaml_path)



