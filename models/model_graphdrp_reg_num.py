import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool
import clip
from collections import OrderedDict
from utils import num2english


'''
这次打算尝试  Text 改成 包含Drug 和 Cell 的 IC50 这种
为之后统一大模型打基础

'''
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
   
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class GNN(nn.Module):
    def __init__(self,
                 input,
                 output,
                 gnn_type,
                 heads=1,
                 dropout=0.2,
                 feature_pre_dropout=0,
                 activate_func='relu'):
        super(GNN, self).__init__()

        self.gnn_type = gnn_type
        
        if feature_pre_dropout>0:
            self.pre_dropout = nn.Dropout(feature_pre_dropout)
            
        if self.gnn_type == 'GINConvNet':
            nn_core = Sequential(Linear(input, output),
                                 ReLU(), Linear(output, output))
            self.gnn = GINConv(nn_core)
            self.bn = torch.nn.BatchNorm1d(output)
        elif self.gnn_type == 'GCNConv':
            self.gnn = GCNConv(input, output)
        elif self.gnn_type == 'GATConv':
            self.gnn = GATConv(input, output, heads=heads, dropout=dropout)

        if activate_func == 'relu':
            self.activate_func = nn.ReLU()
        elif activate_func == 'elu':
            self.activate_func = nn.ELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(self, 'pre_dropout'):
            x = self.pre_dropout(x)
        
        if self.gnn_type == 'GINConvNet':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)
            x = self.bn(x)
        elif self.gnn_type == 'GCNConv':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)
        elif self.gnn_type == 'GATConv':
            x = self.gnn(x, edge_index)
            x = self.activate_func(x)

        data.x = x

        return data


class Drug(nn.Module):
    def __init__(self,
                 module_name,
                 input_drug_feature_dim,
                 output_drug_feature_dim,
                 layer_num,
                 graph_pooling,
                 linear_layers,
                 gnn_layers,
                 dropout):
        super(Drug, self).__init__()

        assert len(
            gnn_layers) == layer_num, 'Number of layer is not same as hyperparameter list.'
        assert graph_pooling in [
            'add', 'max', 'mean', 'max_mean'], 'The type of graph pooling is not right.'

        self.gnn_layers = gnn_layers
        self.linear_layers = linear_layers
        self.graph_pooling = graph_pooling
        self.backbone = nn.Sequential()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        for index, params in enumerate(gnn_layers):
            if module_name[index] == "GATConv":
                self.backbone.add_module(
                    '{0}-{1}'.format(module_name[index], index), GNN(params['intput'], params['output'], module_name[index], heads=params['heads'], dropout=params['dropout'], feature_pre_dropout=params['feature_pre_dropout']))
            else:
                self.backbone.add_module(
                    '{0}-{1}'.format(module_name[index], index), GNN(params['intput'], params['output'], module_name[index]))

        if linear_layers:
            self.linears = nn.Sequential()

            for idx, linear_parameter in enumerate(linear_layers):

                if linear_parameter['operate_name'] == 'linear':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

                elif linear_parameter['operate_name'] == 'relu':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

                elif linear_parameter['operate_name'] == 'dropout':
                    self.linears.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

    def forward(self, data):

        data = self.backbone(data)
        x, batch = data.x, data.batch

        if self.graph_pooling == "add":
            x = global_add_pool(x, batch)
        if self.graph_pooling == "max":
            x = gmp(x, batch)
        if self.graph_pooling == "mean":
            x = gap(x, batch)
        if self.graph_pooling == "max_mean":
            x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.linears(x)

        return x


class Cell(nn.Module):
    def __init__(self,
                 input_cell_feature_dim,
                 output_cell_feature_dim,
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
        self.fc_1 = nn.Linear(channel*cell_feature_dim,
                              output_cell_feature_dim)

    def forward(self, x):

        x = self.backbone(x)
        x = x.view(-1, x.shape[1] * x.shape[2])

        x = self.fc_1(x)
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


class GraphDRP(torch.nn.Module):
    def __init__(self, config):
        super(GraphDRP, self).__init__()
        clip.available_models()
        self.config = config

        # self.drug_module
        self.init_drug_module(self.config['model']['drug_module'])

        # self.cell_module
        self.init_cell_module(self.config['model']['cell_module'])

        # self.fusion_module
        self.init_fusion_module(self.config['model'])
        
        self.loss_fn = nn.MSELoss()
        self.loss_fn_l1 = nn.L1Loss(reduction='mean')
        
        self.loss_domain = torch.nn.NLLLoss()
        
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=0.0, reduction='mean')
        self.memo = {}
        
        # self.logit_scale = 0.1
        
        vocab_size = 50000
        transformer_width = 256
        # self.context_length = context_length
        self.context_length = 300
        transformer_width = 128
        transformer_layers = 3
        transformer_heads = 8
        embed_dim = 128
        
        # test encode
        self.token_embedding_num = nn.Embedding(vocab_size, transformer_width).to(config['cuda_name'])
        self.positional_embedding_num = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final_num = LayerNorm(transformer_width)
        
        self.transformer_num = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.text_projection_num = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()
        

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.to(self.config['cuda_name'])
    
    def initialize_parameters(self):
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

        if self.transformer_num is not None:
            nn.init.normal_(self.text_projection_num, std=self.transformer_num.width ** -0.5)
  
    def get_static(self,cell_name_mp):
        self.cell_name_mp = cell_name_mp
        
    def init_drug_module(self, config):
        module_name = config['module_name']
        input_drug_feature_dim = config['input_drug_feature_dim']
        layer_num = config['layer_num']
        graph_pooling = config['graph_pooling']
        dropout = config['dropout']
        output_drug_feature_dim = config['output_drug_feature_dim']
        linear_layers = config['linear_layers'] if config.get(
            'linear_layers') else None
        gnn_layers = config['gnn_layers']

        self.drug_module = Drug(module_name,
                                input_drug_feature_dim,
                                output_drug_feature_dim,
                                layer_num,
                                graph_pooling,
                                linear_layers,
                                gnn_layers,
                                dropout)

    def init_cell_module(self, config):
        input_cell_feature_dim = config['input_cell_feature_dim']
        module_name = config['module_name']
        fc_1_dim = config['fc_1_dim']
        layer_num = config['layer_num']
        dropout = config['transformer_dropout'] if config.get(
            'transformer_dropout') else 0
        layer_hyperparameter = config['layer_hyperparameter']
        output_cell_feature_dim = config['output_cell_feature_dim']

        self.cell_module = Cell(input_cell_feature_dim,
                                output_cell_feature_dim,
                                module_name,
                                fc_1_dim,
                                layer_num,
                                dropout,
                                layer_hyperparameter)

    def init_fusion_module(self, config):
        input_dim = [config['drug_module']['output_drug_feature_dim'],
                     config['cell_module']['output_cell_feature_dim']]

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
        fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
        
        return pred_y, fusion_features
    
    
    # def forward(self, data):
        
    #     device = data.x.device
    #     x_drug = self.drug_module(data)
    #     x_cell = self.cell_module(data.target[:, None, :])
    #     pred_y, fusion_features = self.fusion_module(x_drug, x_cell)
        
        
    #     # descriptions_text = []
    #     # descriptions_number = []
    #     descriptions = []
    #     for index, item in enumerate(data.y):
    #         # pdb.set_trace()
            

    #         # descriptions_number.append(des)
            
    #         des = "The drug response value between " + data.smiles[index] + " and "+ data.cell_name[index] +" is " + "zero point " + num2english(item.item())
            
    #                     # des =  str(translateNumberToEnglish(item.item()))
            
    #         # des = "zero point " + num2english(item.item())
            
            
    #         descriptions.append(des)
            
    #         # continue
        
    #     # text = clip.tokenize(descriptions_text,context_length=300).to(device)
    #     number = clip.tokenize(descriptions,context_length=300).to(device)
    #     # 
      
    #     # text_features = self.encode_text(text)
    #     number_features = self.encode_num(number)
        
    #     # sentence_features = torch.cat((text_features,number_features),axis=1)
        
    #     # sentence_features = self.transformer_fusion(sentence_features)
    #     # normalized features
    #     fusion_features = fusion_features / fusion_features.norm(dim=1, keepdim=True)
    #     number_features = number_features / number_features.norm(dim=1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_dc = logit_scale * fusion_features @ number_features.t()
    #     # logits_per_dc =  fusion_features @ text_features.t()
    #     logits_per_text = logits_per_dc.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_dc, logits_per_text
    
    
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
    
    