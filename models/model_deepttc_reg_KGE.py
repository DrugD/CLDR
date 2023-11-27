import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool
from nlp_encoder import  *
from utils import num2english
import clip
from model_helper import Embeddings,Encoder_MultipleLayers


class Drug(nn.Module):
    def __init__(self,
                 config):
        super(Drug, self).__init__()
       
        
        self.emb = Embeddings(config['input_drug_feature_dim'],
                         config['gnn_layers']['embed_dim'],
                         50,
                         config['dropout'])

        self.encoder = Encoder_MultipleLayers(config['layer_num'],
                                         config['gnn_layers']['embed_dim'],
                                         config['gnn_layers']['intermediate_dim'],
                                         config['gnn_layers']['head'],
                                         config['gnn_layers']['attention_probs_dropout'],
                                         config['gnn_layers']['hidden_dropout'],)
        self.DeepTTC_encode = torch.load("/home/lk/project/MSDA/data/process/GDSC2_dataset/DeepTTC_encode.pth")

  
    def forward(self, data):

        DeepTTC_drug_encode = []
        DeepTTC_drug_encode_mask = []
        
        for smile in data.smiles:
            DeepTTC_drug_encode.append(self.DeepTTC_encode[smile]['DeepTTC_drug_encode'].to(data.x.device))
            DeepTTC_drug_encode_mask.append(self.DeepTTC_encode[smile]['DeepTTC_drug_encode_mask'].to(data.x.device))
        
        v = [torch.cat(DeepTTC_drug_encode, dim=0), torch.cat(DeepTTC_drug_encode_mask, dim=0)]
        e = v[0].long()
        e_mask = v[1].long()
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]



class Cell(nn.Module):
    def __init__(self,
                input_cell_feature_dim,
                output_cell_feature_dim,
                module_name,
                linear_layers):
        super(Cell, self).__init__()

        self.module_name = module_name

        self.backbone = nn.Sequential()
        self.relu = nn.ReLU()
        
        if linear_layers:
            self.backbone = nn.Sequential()

            for idx, linear_parameter in enumerate(linear_layers):

                if linear_parameter['operate_name'] == 'linear':
                    self.backbone.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

                elif linear_parameter['operate_name'] == 'relu':
                    self.backbone.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

                elif linear_parameter['operate_name'] == 'dropout':
                    self.backbone.add_module(
                        '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

    def forward(self, x):
        x = x.squeeze()
        x = self.backbone(x)
        return x


class Fusion(nn.Module):
    def __init__(self,
                module_name,
                linear_layers,
                cnn_layers,
                fc_1,
                fusion_mode):
        super(Fusion, self).__init__()

        self.fusion_mode = fusion_mode
        self.relu = nn.ReLU()
        
        self.backbone = nn.Sequential()

        for idx, linear_parameter in enumerate(linear_layers):

            if linear_parameter['operate_name'] == 'linear':
                self.backbone.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), torch.nn.Linear(linear_parameter['param'][0], linear_parameter['param'][1]))

            elif linear_parameter['operate_name'] == 'relu':
                self.backbone.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), self.relu)

            elif linear_parameter['operate_name'] == 'dropout':
                self.backbone.add_module(
                    '{0}-{1}'.format(linear_parameter['operate_name'], idx), nn.Dropout(linear_parameter['param']))

            elif linear_parameter['operate_name'] == 'conv1d':
                self.backbone.add_module('CNN1d-{0}_{1}_{2}'.format(idx), nn.Conv1d(in_channels=linear_parameter['cnn_channels'][0],
                            out_channels=linear_parameter['cnn_channels'][1],
                            kernel_size=linear_parameter['kernel_size']))
            
            elif linear_parameter['operate_name'] == 'maxpool1d':
                self.backbone.add_module('Maxpool-{0}'.format(idx),nn.MaxPool1d(
                                        linear_parameter['param']))
                        
 
        
    def forward(self, drug, cell):

        if self.fusion_mode == "concat":
            x = torch.cat((drug, cell), 1)
            
        x_features = None
        # pdb.set_trace()
        for ops in self.backbone:
            if 'out_features' in ops.__dict__.keys() and ops.__dict__['out_features'] == 256:
                x = ops(x)
                x_features = x
            
            else:
                x = ops(x)
               
        return x, x_features


class DeepTTC(torch.nn.Module):
    def __init__(self, config):
        super(DeepTTC, self).__init__()

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
        self.drug_module = Drug(config)

    def init_cell_module(self, config):
        
        module_name = config['module_name']
        input_cell_feature_dim = config['input_cell_feature_dim']
        output_cell_feature_dim = config['output_cell_feature_dim']
        linear_layers = config['linear_layers'] if config.get(
            'linear_layers') else None
        
        self.cell_module = Cell(input_cell_feature_dim,
                                output_cell_feature_dim,
                                module_name,
                                linear_layers)

    def init_fusion_module(self, config):
        module_name = config['fusion_module']['module_name']
        
        linear_layers = config['fusion_module']['linear_layers']
        cnn_layers = config['fusion_module']['cnn_layers'] if  config['fusion_module'].get('cnn_layers') else None

        fusion_mode = config['fusion_module']['fusion_mode']
        fc_1 = config['fusion_module']['fc_1'] if  config['fusion_module'].get('fc_1') else None
        
        self.fusion_module = Fusion(module_name,
                                    linear_layers,
                                    cnn_layers,
                                    fc_1,
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
    
    

