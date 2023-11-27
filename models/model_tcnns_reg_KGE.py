import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
import pdb
import numpy as np
from model_helper import Embeddings,Encoder_MultipleLayers
from nlp_encoder import  *
from utils import num2english
import clip


class Drug(nn.Module):
    def __init__(self,
                 input_drug_feature_dim,
                 input_drug_feature_channel,
                 layer_hyperparameter,
                 layer_num):
        super(Drug, self).__init__()

        assert len(
            layer_hyperparameter) == layer_num, 'Number of layer is not same as hyperparameter list.'

        self.input_drug_feature_channel = input_drug_feature_channel
        input_channle = input_drug_feature_channel
        drug_feature_dim = input_drug_feature_dim

        self.backbone = nn.Sequential()

        for index, channel in enumerate(layer_hyperparameter['cnn_channels']):

            self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(index, input_channle, channel), nn.Conv1d(in_channels=input_channle,
                                                                                                          out_channels=channel,
                                                                                                          kernel_size=layer_hyperparameter['kernel_size'][index]))
            self.backbone.add_module('ReLU-{0}'.format(index), nn.ReLU())
            self.backbone.add_module('Maxpool-{0}'.format(index), nn.MaxPool1d(
                layer_hyperparameter['maxpool1d'][index]))
            input_channle = channel
            drug_feature_dim = int(((
                drug_feature_dim-layer_hyperparameter['kernel_size'][index]) + 1)/layer_hyperparameter['maxpool1d'][index])

        self.drug_output_feature_channel = channel
        self.drug_output_feature_dim = drug_feature_dim


    
        self.tCNNs_encode = torch.load("/home/lk/project/MSDA/data/process/GDSC2_dataset/tcnns_encode.pth")


     


    def forward(self, data):
        tCNNs_drug_matrix = []
        # pdb.set_trace()
        
        for smile in data.smiles:
            tCNNs_drug_matrix.append(torch.tensor(self.tCNNs_encode[smile]['tCNNs_drug_matrix']).to(data.x.device))
        
        # pdb.set_trace()
        x = torch.stack(tCNNs_drug_matrix, dim=0)
        
        
        if x.shape[1] != self.input_drug_feature_channel:
            x = torch.cat((torch.zeros(
                (x.shape[0], self.input_drug_feature_channel - x.shape[1], x.shape[2]), dtype=torch.float).cuda(), x), 1)
        
        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        return x





class Cell(nn.Module):
    def __init__(self,
                 input_cell_feature_dim,
                 module_name,
                 fc_1_dim,
                 layer_num,
                 dropout,
                 layer_hyperparameter):
        super(Cell, self).__init__()

        self.module_name = module_name

        assert len(
            layer_hyperparameter) == layer_num, 'Number of layer is not same as hyperparameter list.'

        self.backbone = nn.Sequential()

        input_channle = 1
        cell_feature_dim = input_cell_feature_dim

        for index, channel in enumerate(layer_hyperparameter['cnn_channels']):

            self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(index, input_channle, channel), nn.Conv1d(in_channels=input_channle,
                                                                                                          out_channels=channel,
                                                                                                          kernel_size=layer_hyperparameter['kernel_size'][index]))
            self.backbone.add_module('ReLU-{0}'.format(index), nn.ReLU())
            self.backbone.add_module('Maxpool-{0}'.format(index), nn.MaxPool1d(
                layer_hyperparameter['maxpool1d'][index]))

            input_channle = channel
            cell_feature_dim = int(((
                cell_feature_dim-layer_hyperparameter['kernel_size'][index]) + 1)/layer_hyperparameter['maxpool1d'][index])

        self.cell_output_feature_channel = channel
        self.cell_output_feature_dim = cell_feature_dim

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        return x


class Fusion(nn.Module):
    def __init__(self,
                 input_dim,
                 fc_1_dim,
                 fc_2_dim,
                 fc_3_dim,
                 dropout,
                 fusion_mode):
        super(Fusion, self).__init__()

        self.fusion_mode = fusion_mode

        if fusion_mode == "concat":
            input_dim = input_dim[0]+input_dim[1]
            self.fc1 = nn.Linear(input_dim, fc_1_dim)

        self.fc2 = nn.Linear(fc_1_dim, fc_2_dim)
        self.fc3 = nn.Linear(fc_2_dim, fc_3_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x_feature = self.fc2(x)
        x = self.relu(x_feature)
        x = self.dropout(x)

        x = self.fc3(x)
        x = nn.Sigmoid()(x)

        return x, x_feature


class tCNNs(torch.nn.Module):
    def __init__(self, config):
        super(tCNNs, self).__init__()

        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])

        vocab_size = 50000
        transformer_width = 256
        # self.context_length = context_length
        self.context_length = 300
        self.context_num_length = 100
        transformer_width = 128
        transformer_layers = 3
        transformer_heads = 8
        embed_dim = 128
        
        # test encode
        self.token_embedding = nn.Embedding(vocab_size, transformer_width).to(config['cuda_name'])
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        
        self.token_embedding_num = nn.Embedding(vocab_size, transformer_width).to(config['cuda_name'])
        self.positional_embedding_num = nn.Parameter(torch.empty(self.context_num_length, transformer_width))
        self.ln_final_num = LayerNorm(transformer_width)
        
        self.transformer_num = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask_num()
        )

        self.text_projection_num = nn.Parameter(torch.empty(transformer_width, embed_dim))


        self.transformer_fusion = MLP(embed_dim*2, 512, embed_dim*2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.to(self.config['cuda_name'])
    
    def build_attention_mask_num(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_num_length, self.context_num_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.to(self.config['cuda_name'])
    
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
        nn.init.normal_(self.token_embedding_num.weight, std=0.02)
        nn.init.normal_(self.positional_embedding_num, std=0.01)
        proj_std = (self.transformer_num.width ** -0.5) * ((2 * self.transformer_num.layers) ** -0.5)
        attn_std = self.transformer_num.width ** -0.5
        fc_std = (2 * self.transformer_num.width) ** -0.5
        for block in self.transformer_num.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection_num, std=self.transformer_num.width ** -0.5)
  
    def init_drug_module(self, config):
        input_drug_feature_dim = config['input_drug_feature_dim']
        input_drug_feature_channel = config['input_drug_feature_channel']
        layer_hyperparameter = config['layer_hyperparameter']
        layer_num = config['layer_num']

        self.drug_module = Drug(input_drug_feature_dim,
                                input_drug_feature_channel,
                                layer_hyperparameter,
                                layer_num)

    def init_cell_module(self, config):
        input_cell_feature_dim = config['input_cell_feature_dim']
        module_name = config['module_name']
        fc_1_dim = config['fc_1_dim']
        layer_num = config['layer_num']
        dropout = config['transformer_dropout'] if config.get(
            'transformer_dropout') else 0
        layer_hyperparameter = config['layer_hyperparameter']

        self.cell_module = Cell(input_cell_feature_dim,
                                module_name,
                                fc_1_dim,
                                layer_num,
                                dropout,
                                layer_hyperparameter)

    def init_fusion_module(self, config):
        input_dim = [self.drug_module.drug_output_feature_dim * self.drug_module.drug_output_feature_channel,
                     self.cell_module.cell_output_feature_dim * self.cell_module.cell_output_feature_channel]
       
        fc_1_dim = config['fusion_module']['fc_1_dim']
        fc_2_dim = config['fusion_module']['fc_2_dim']
        fc_3_dim = config['fusion_module']['fc_3_dim']
        dropout = config['fusion_module']['dropout']
        fusion_mode = config['fusion_module']['fusion_mode']
        
        self.fusion_module = Fusion(input_dim,
                                    fc_1_dim,
                                    fc_2_dim,
                                    fc_3_dim,
                                    dropout,
                                    fusion_mode)

    def encode_text(self, text):
        
        # pdb.set_trace()
        x = self.token_embedding(text.unsqueeze(2)).squeeze()  # [batch_size, n_ctx, d_model]
        

        
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
            
    def encode_num(self, text):
        
        # pdb.set_trace()
        x = self.token_embedding_num(text.unsqueeze(2)).squeeze()  # [batch_size, n_ctx, d_model]
        

        
        x = x + self.positional_embedding_num
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_num(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_num(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection_num

        return x
 
    def forward(self, data):
        
        device = data.x.device
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])
        pred_y, fusion_features = self.fusion_module(x_drug, x_cell)
        
        
        descriptions_text = []
        descriptions_number = []
        for index, item in enumerate(data.y):
            # pdb.set_trace()
            
            # des =  str(translateNumberToEnglish(item.item()))
            
            des = "zero point " + num2english(item.item())
            descriptions_number.append(des)
            
            des = "The drug response value between " + data.smiles[index] + " and "+ data.cell_name[index] +" is "
            descriptions_text.append(des)
            
            # continue
        
        text = clip.tokenize(descriptions_text,context_length=300).to(device)
        number = clip.tokenize(descriptions_number,context_length=100).to(device)
        # 
      
        text_features = self.encode_text(text)
        number_features = self.encode_num(number)
        
        sentence_features = torch.cat((text_features,number_features),axis=1)
        
        sentence_features = self.transformer_fusion(sentence_features)
        # normalized features
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        sentence_features = sentence_features / sentence_features.norm(dim=1, keepdim=True)

        # pdb.set_trace()
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_dc = logit_scale * fusion_features @ sentence_features.t()
        # logits_per_dc =  fusion_features @ text_features.t()
        logits_per_text = logits_per_dc.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_dc, logits_per_text
    
    
        # return  fusion_features, text_features
            
            
            # reg_loss = self.loss_fn(pred_y.float(), data.y.view(-1, 1).float().to(device))
            
            # rank_loss = self.ranking_loss(data.y.view(-1, 1).float().to(device) , pred_y.float(), torch.ones_like(pred_y.float()))

            # main_loss = reg_loss * 0.9 + rank_loss * 0.1

            # return main_loss, fusion_feature
        
    def infer(self, data):
        device = data.x.device
        x_drug = self.drug_module(data)
        x_cell = self.cell_module(data.target[:, None, :])
        pred_y, fusion_features = self.fusion_module(x_drug, x_cell)
        
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_dc = logit_scale * fusion_features @ text_features.t()
        # logits_per_text = logits_per_dc.t()

        # shape = [global_batch_size, global_batch_size]
        return fusion_features
    
    