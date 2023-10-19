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


# training function at each epoch
# training function at each epoch
import torch
import os
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

    for ic50 in range(0,1000,1):
        des = "zero point " + num2english(ic50/1000)
        descriptions.append(des)

    text = clip.tokenize(descriptions,context_length=300).to(device)
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
            gt_sentence_features = gt_sentence_features.repeat(drug_cell_logits.size()[0],1,1)
            
            fusion_features = model.infer(data_)
            fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
            fusion_features = fusion_features.unsqueeze(2)
            
            pred = (model.logit_scale.exp()*torch.bmm(gt_sentence_features,fusion_features)).squeeze()
            
            pred = (torch.argmax(pred,1)/1000).cpu().view(-1, 1)
            total_preds = torch.cat((total_preds, pred), 0)
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


def main(config, yaml_path):



    model = GraphDRP(config)
    # model.load_state_dict(torch.load(
    #     "/home/lk/project/DALLE24Drug/CLIP4Drug/CLIP_DRP/exp/GAT_GCN_number/text_num_GDSCv2__20231011195851/GraphDRP.model", map_location=torch.device(device)), strict=True)

    model.to(device)


    train_batch = config["batch_size"]["train"]
    val_batch = config["batch_size"]["val"]
    test_batch = config["batch_size"]["test"]
    lr = config["lr"]
    num_epoch = config["num_epoch"]
    log_interval = config["log_interval"]

    work_dir = config["work_dir"]

    date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""

    date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
    work_dir = "./exp/" + config['marker'] + "/" + work_dir + "_"  +  date_info
    
    if not os.path.exists("./exp/" + config['marker']):
        os.mkdir("./exp/" + config['marker'])
        
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    copyfile(yaml_path, work_dir + "/")
    model_st = config["model_name"]
    
    train_dataset, val_dataset, test_dataset = load(config['dataset_type'])

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


    for epoch in tqdm(range(num_epoch+1)):

        reg_loss = train(
            model, device,  train_loader, optimizer, epoch + 1, log_interval, config
        )

        G, P = predicting(model, device, val_loader, "val", config)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]
        # pdb.set_trace()

        # if ret[1] < best_mse and epoch>10:
        G_test, P_test = predicting(
            model, device, test_loader, "test", config)

        ret_test = [
            rmse(G_test, P_test),
            mse(G_test, P_test),
            pearson(G_test, P_test),
            spearman(G_test, P_test),
            rankingLossFunc(torch.tensor(G_test), torch.tensor(
                P_test), torch.ones_like(torch.tensor(P_test))).item()
        ]
        # print(ret_test)

   

  

        ret_test = [
                rmse(G_test, P_test),
                mse(G_test, P_test),
                pearson(G_test, P_test),
                spearman(G_test, P_test),
                rankingLossFunc(torch.tensor(G_test) , torch.tensor(P_test), torch.ones_like(torch.tensor(P_test))).item()
            ]
        
  

        train_losses.append(reg_loss)
        val_losses.append(ret_test[1])
        val_pearsons.append(ret_test[2])

        # draw_sort_pred_gt(P, G, title=work_dir + "/val_" +str(epoch))

        draw_sort_pred_gt(
            P_test, G_test, title=work_dir + "/test_" + str(epoch))

        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
        
            with open(result_file_name, "a") as f:
                f.write("\n " + str(epoch))
                f.write("\n rmse:"+str(ret_test[0]))
                f.write("\n mse:"+str(ret_test[1]))
                f.write("\n pearson:"+str(ret_test[2]))
                f.write("\n spearman:"+str(ret_test[3]))
                f.write("\n rankingloss:"+str(ret_test[4])+"\n")
                
            best_epoch = epoch + 1
            best_mse = ret_test[1]
            best_pearson = ret_test[2]
            print(
                " rmse improved at epoch ",
                best_epoch,
                "; best_mse:",
                best_mse,
                model_st,
            )

        else:
            print(
                " no improvement since epoch ",
                best_epoch,
                "; best_mse, best pearson:",
                best_mse,
                best_pearson,
                model_st,
            )

        draw_loss(train_losses, val_losses, loss_fig_name)
        draw_pearson(val_pearsons, pearson_fig_name)


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
