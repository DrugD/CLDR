import sys

sys.path.insert(0,"./")



import math
import pdb
import re
import os
import torch.nn.functional as F
from utils import copyfile
import random
import datetime
from utils import *

# from models.model_graphdrp import GraphDRP
# from models.model_graphdrp_reg_num2 import GraphDRP
from models.model_transedrp_reg_KGE import TransEDRP

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()

PRECISION = 1




def generate_samples(model, data, start, end):
    # data: list, consist of [drug smile, cell line, ic50]
    descriptions = []
    assert end - start == int('1'+'0'*PRECISION)
    
    # if model.training:    
    # for ic50 in range(start,end,1):
    for idx, ic50 in enumerate(range(0,int('1'+'0'*PRECISION),1)):
        # 
        # pdb.set_trace()
        # print(ic50)
        des = "zero point" + num2english(ic50/int('1'+'0'*PRECISION), PRECISION)
        
        descriptions.append(des)
            # pdb.set_trace()
    text = clip.tokenize(descriptions,context_length=100).to(device)
    # pdb.set_trace()
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
            
            fusion_features = model.infer(data)
            
            number_features = generate_samples(model, data, 0, int('1'+'0'*PRECISION))
            # gt_sentence_features = gt_sentence_features/gt_sentence_features.norm(dim=1,keepdim=True)
            
            preds = torch.Tensor()
            for data_item_index in range(len(data)):
                des = "The drug response value between " + data.smiles[data_item_index] + " and "+ data.cell_name[data_item_index] +" is "
                text = clip.tokenize([des,des],context_length=300).to(device)
                
                text_features = model.encode_text(text)[0]
                logits = torch.Tensor()
                for interval_index in range(10):
                    ten_num = int(int('1'+'0'*PRECISION)/10)
                    sentence_features = torch.cat((text_features.repeat(ten_num,1),number_features[interval_index*ten_num:(interval_index+1)*ten_num]),axis=1)
            
                    sentence_features = model.transformer_fusion(sentence_features)
                    sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

                    

                    
                    logit = model.logit_scale * fusion_features[data_item_index].unsqueeze(0) @ sentence_features.t()
                    logits = torch.cat((logits,logit.cpu()),1)
                
                
                pred = (torch.argmax(logits,1)/int('1'+'0'*PRECISION)).view(-1, 1)
                preds = torch.cat((preds,pred),0)
                
            # pdb.set_trace() 

            total_preds = torch.cat((total_preds, preds), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        
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


def main(config, yaml_path, infer_model):



    model = TransEDRP(config)

    model.load_state_dict(torch.load(
        infer_model, map_location='cpu'), strict=True)
    
    # device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    train_batch = config["batch_size"]["train"]
    val_batch = config["batch_size"]["val"]
    test_batch = config["batch_size"]["test"]
    lr = config["lr"]
    num_epoch = config["num_epoch"]
    log_interval = config["log_interval"]

    work_dir = config["work_dir"]

    date_info = ("_infer" + dateStr()) if config["work_dir"] != "test" else ""

    # date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
    work_dir = "./exp/" + config['marker'] + "/" + work_dir + "_"  +  date_info
    
    if not os.path.exists("./exp/" + config['marker']):
        os.mkdir("./exp/" + config['marker'])
        
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    copyfile(yaml_path, work_dir + "/")
    model_st = config["model_name"]

    
    trainval_dataset, test_dataset = load(config)
    
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,\
        [
            int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val']))),\
            len(trainval_dataset)-int(len(trainval_dataset)*(config["dataset_type"]['train']/(config["dataset_type"]['train']+config["dataset_type"]['val'])))
        ])
        
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size']['train'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size']['val'], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size']['test'], shuffle=False, drop_last=False)


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler = None
   

    best_mse = 9999
    best_pearson = 1
    best_epoch = -1

    model_file_name = work_dir + "/" + model_st + ".model"
    result_file_name = work_dir + "/" + model_st + ".csv"
    loss_fig_name = work_dir + "/" + model_st + "_loss"
    pearson_fig_name = work_dir + "/" + model_st + "_pearson"

    train_losses = []
    val_losses = []
    val_pearsons = []

    rankingLossFunc = torch.nn.MarginRankingLoss(
        margin=0.0, reduction='mean')


    # pdb.set_trace()
    # torch.save(model.state_dict(), model_file_name)
    # model.to(device)
    # pdb.set_trace()
    # model.load_state_dict(torch.load(infer_model, map_location='cpu'))
    # pdb.set_trace()
    # G, P = predicting(model, device, val_loader, "val", config)
    # ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]
    # pdb.set_trace()

    # if ret[1] < best_mse and epoch>10:
    G_test, P_test = predicting(
        model, device, test_loader, "test", config)
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
    
    # train_losses.append(0)
    # val_losses.append(ret[1])
    # val_pearsons.append(ret[2])

    # draw_sort_pred_gt(P, G, title=work_dir + "/val_" +str(epoch))

    # draw_sort_pred_gt(
    #     P_test, G_test, title=work_dir + "/test_" + str(epoch))

    # if ret[1] < best_mse:
    #     torch.save(model.state_dict(), model_file_name)
    
    #     with open(result_file_name, "a") as f:
    #         f.write("\n " + str(epoch))
    #         f.write("\n rmse:"+str(ret_test[0]))
    #         f.write("\n mse:"+str(ret_test[1]))
    #         f.write("\n pearson:"+str(ret_test[2]))
    #         f.write("\n spearman:"+str(ret_test[3]))
    #         f.write("\n rankingloss:"+str(ret_test[4])+"\n")
            
    #     best_epoch = epoch + 1
    #     best_mse = ret[1]
    #     best_pearson = ret[2]
    #     print(
    #         " rmse improved at epoch ",
    #         best_epoch,
    #         "; best_mse:",
    #         best_mse,
    #         model_st,
    #     )

    # else:
    #     print(
    #         " no improvement since epoch ",
    #         best_epoch,
    #         "; best_mse, best pearson:",
    #         best_mse,
    #         best_pearson,
    #         model_st,
    #     )

    # draw_loss(train_losses, val_losses, loss_fig_name)
    # draw_pearson(val_pearsons, pearson_fig_name)


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
    seed_torch(config["seed"])

    cuda_name = config["cuda_name"]

    print("CPU/GPU: ", torch.cuda.is_available())

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    main(config, yaml_path, infer_model)
