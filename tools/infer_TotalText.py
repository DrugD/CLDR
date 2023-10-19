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

import datetime
import argparse
import random

from utils import copyfile

# training function at each epoch
CUDA_LAUNCH_BLOCKING=1



def generate_samples(model, data, start, end, index):
    # data: list, consist of [drug smile, cell line, ic50]
    descriptions = []
    assert end - start == 1000
    # pdb.set_trace()
    if model.training:
    # for ic50 in range(start,end,1):
        for ic50 in index:
            des = "zero point " + num2english(ic50/1000)
            descriptions.append(des)
        model.train()
        text = clip.tokenize(descriptions,context_length=300).to(device)
        text_features = model.encode_num(text)
        
    else:
        for ic50 in range(start,end,1):
            des = "zero point " + num2english(ic50/1000)
            descriptions.append(des)
            
        with torch.no_grad():
            model.eval()
            text = clip.tokenize(descriptions,context_length=300).to(device)
            text_features = model.encode_num(text)
    
 
    return text_features

# def predicting(model, device, loader, loader_type, args):
#     model.eval()
#     total_preds = torch.Tensor()
#     total_labels = torch.Tensor()
#     print("Make prediction for {} samples...".format(len(loader.dataset)))
#     with torch.no_grad():
#         for batch_idx, data in tqdm(enumerate(loader)):
#             data = data.to(device)
#             data_ = deepcopy(data)
            
#             _, fusion_features = model(data)
            
#             gt_sentence_features = generate_samples(model, data, 0, 1000, (data.y*1000).int().cpu().numpy())
#             # gt_sentence_features = gt_sentence_features/gt_sentence_features.norm(dim=1,keepdim=True)
#             gt_sentence_features = gt_sentence_features.repeat(data.y.size()[0],1,1)
#             # gt_sentence_features = gt_sentence_features.unsqueeze(1)
#             fusion_features = fusion_features.unsqueeze(2)
#             # pdb.set_trace()
            
#             pred = (torch.bmm(gt_sentence_features,fusion_features)).squeeze()
#             pred = (torch.argmax(pred,1)/1000).cpu().view(-1, 1)
#             total_preds = torch.cat((total_preds, pred), 0)
#             total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader)):
            data = data.to(device)
            # data_ = deepcopy(data)
            
            pred, _ = model(data)
            
            # gt_sentence_features = generate_samples(model, data, 0, 1000, (data.y*1000).int().cpu().numpy())
            # # gt_sentence_features = gt_sentence_features/gt_sentence_features.norm(dim=1,keepdim=True)
            # gt_sentence_features = gt_sentence_features.repeat(data.y.size()[0],1,1)
            # # gt_sentence_features = gt_sentence_features.unsqueeze(1)
            # fusion_features = fusion_features.unsqueeze(2)
            # # pdb.set_trace()
            
            # pred = (torch.bmm(gt_sentence_features,fusion_features)).squeeze()
            # pred = (torch.argmax(pred,1)/1000).cpu().view(-1, 1)
            total_preds = torch.cat((total_preds, pred.cpu().view(-1, 1)), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


            
# def predicting(model, device, loader, loader_type, args):
#     model.eval()
#     total_preds = torch.Tensor()
#     total_labels = torch.Tensor()
#     print("Make prediction for {} samples...".format(len(loader.dataset)))
#     with torch.no_grad():
#         for batch_idx, data in tqdm(enumerate(loader)):
#             data = data.to(device)
#             data_ = deepcopy(data)
            
#             output, fusion_features = model(data)
            
#             gt_sentence_features = generate_samples(model, data, 0, 1000, (data.y*1000).int().cpu().numpy())
#             # gt_sentence_features = gt_sentence_features/gt_sentence_features.norm(dim=1,keepdim=True)
#             gt_sentence_features = gt_sentence_features.repeat(data.y.size()[0],1,1)
#             # gt_sentence_features = gt_sentence_features.unsqueeze(1)
#             fusion_features = fusion_features.unsqueeze(2)
#             # pdb.set_trace()
            
#             pred = (torch.bmm(gt_sentence_features,fusion_features)).squeeze()
#             pred = (torch.argmax(pred,1)/1000).cpu().view(-1, 1)
            
#             # only MSE
#             # pred = output.cpu().view(-1, 1)
            
#             # only CLIP
            
            
#             # two
#             pred = (pred+output.cpu().view(-1, 1))/2
            
#             total_preds = torch.cat((total_preds, pred), 0)
#             total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )


def main(model, config, yaml_path, show_type):

    val_batch = config["batch_size"]["val"]
    test_batch = config["batch_size"]["test"]

    cuda_name = config["cuda_name"]
    work_dir = config["work_dir"]

    date_info = ("_infer_" + dateStr()) if config["work_dir"] != "test" else ""
    work_dir = "./exp/" + work_dir + date_info
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    copyfile(yaml_path, work_dir + "/")


    # _, val_data, test_data = load(config=config["dataset_type"])
    trainval_dataset, test_dataset = load(config)

    # make data PyTorch mini-batch processing ready
    
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,\
        [
            int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val']))),\
            len(trainval_dataset)-int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val'])))
        ])
        
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)
    print("CPU/GPU: ", torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    rankingLossFunc = torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')
   


    epoch='xx'
    

    

    
    if show_type == 'total':
        G, P = predicting(model, device, val_loader, "val", config)
        draw_sort_pred_gt(P, G, title=work_dir + "/val_" +str(epoch))
        
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P),
            rankingLossFunc(torch.tensor(G) , torch.tensor(P), torch.ones_like(torch.tensor(P)))
            ]
        print("Val:",ret)
        
        G_test, P_test = predicting(model, device, test_loader, "test", config)
        draw_sort_pred_gt(P_test, G_test, title=work_dir + "/test_" +str(epoch))
        ret_test = [
            rmse(G_test, P_test),
            mse(G_test, P_test),
            pearson(G_test, P_test),
            spearman(G_test, P_test),
            rankingLossFunc(torch.tensor(G_test) , torch.tensor(P_test), torch.ones_like(torch.tensor(P_test)))
        ]
        print("Test:",ret_test)
        
    elif show_type == 'byclass':
        
        # G, P = predicting(model, device, val_loader, "val", config)
        # draw_sort_pred_gt_classed(P, G, work_dir + "/", val_loader, 'val')
        
        # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P),
        #     rankingLossFunc(torch.tensor(G) , torch.tensor(P), torch.ones_like(torch.tensor(P)))
        #     ]
        # print("Val:",ret)
        
        
        G_test, P_test = predicting(model, device, test_loader, "test", config)
        result_drugs = draw_sort_pred_gt_classed(P_test, G_test, work_dir + "/", test_loader, 'test')
        ret_test = [
            rmse(G_test, P_test),
            mse(G_test, P_test),
            pearson(G_test, P_test),
            spearman(G_test, P_test),
            rankingLossFunc(torch.tensor(G_test) , torch.tensor(P_test), torch.ones_like(torch.tensor(P_test)))
        ]
        print("Test:",ret_test)
        result_drugs['ret_test'] = [ str(x)[:6] for x in ret_test]
        import json
        result_file_name = work_dir + "/" + 'result_drugs' + ".json"
        with open(result_file_name, "a") as f:
            f.write(json.dumps(result_drugs))
    
    
    
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

    modeling = [GraphDRP][
        config["model_type"]
    ]
    model = modeling(config)
    
    model.load_state_dict(torch.load(infer_model))
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    # main(model, config, yaml_path, show_type= 'total')
    main(model, config, yaml_path, show_type= 'byclass')
