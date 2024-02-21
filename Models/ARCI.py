import torch
import torch.nn as nn
import numpy as np
import transformers as ct
import torch.nn.functional as F
import random
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pytorchltr.loss import PairwiseHingeLoss, LambdaNDCGLoss1, PairwiseDCGHingeLoss
import pyhealth.tokenizer
from Models.arki_retain import RETAINLayer, ArkiLinearRetain

from Models import knowledge_graph as kg
from Utils import drug_onthology as do
from typing import Tuple, List, Dict, Optional
from pyhealth.datasets import SampleBaseDataset
import functools
from pyhealth.models.utils import batch_to_multihot

from pytorchltr.evaluation import ndcg
from pyhealth.metrics import ddi_rate_score
from pyhealth.medcode import ATC
#from BaseModel_Custom import BaseModel
from pyhealth.models.base_model import BaseModel
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt



def attention_layer(input_tensor, layer, intent_tensor):
    # Assuming input_tensor has size (B, T, D)

    # Calculate attention weights
    print('input_attention', np.shape(input_tensor), np.shape(intent_tensor))

    attention_weights = F.softmax(layer(torch.cat((input_tensor, intent_tensor),-1)), dim=1)

    # Apply attention to the input tensor
    attended_tensor = torch.sum(attention_weights * input_tensor, dim=1)

    return attended_tensor


class PresRec(BaseModel):
    def __init__(self, dataset: SampleBaseDataset, feature_keys: List[str], label_key: str, mode: str,
                 embedding_dim_temp=128, max_n_codes=49, device=1, n_time_step=3,
                 n_heads=4, negative_ratio=0.1, tmpr=0.8, cl=0.01, ce = 0, interpretability = False):  # n_rx = maximum number of rx codes in each visit,
        super().__init__(dataset, feature_keys, label_key, mode)
        self.feat_tokenizers = {}
        self.number_of_iter = 0
        self.mlml = nn.MultiLabelMarginLoss()
        #print('self_device1', self.device)
        self.embeddings = nn.ModuleDict()
        self.drop = nn.Dropout(0.2)
        self.linear_layers = nn.ModuleDict()
        self.activation = nn.Sigmoid()
        self.label_tokenizer = self.get_label_tokenizer()
        embedding_dim = int(int(embedding_dim_temp/n_heads)*n_heads)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.iter = 0
        self.embedding_dim = embedding_dim
        self.ce_val = ce
        self.ranking_loss = LambdaNDCGLoss1()
        self.interpretability = interpretability
        print(feature_keys)
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Transformer only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                    input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim float and int as input types"
                )

            self.add_feature_transform_layer(feature_key, input_info)
        self.label_size = self.label_tokenizer.get_vocabulary_size()
        self.ddi_adj = torch.nn.Parameter(self.generate_ddi_adj(), requires_grad=False)

        self.recommenders = nn.ModuleDict()
        self.feature_keys = feature_keys
        self.cl = cl
        #self.output_layer_class = nn.Linear(embedding_dim*len(feature_keys)*n_time_step,
        #                                    self.get_output_size(self.get_label_tokenizer()))
        self.output_layer_class = nn.Linear(embedding_dim * len(feature_keys),
                                            self.get_output_size(self.get_label_tokenizer()))
        self.attnetion_based = nn.Linear(embedding_dim * n_time_step * 2,
                                         embedding_dim * n_time_step)

        self.fusion_based = nn.Sequential(nn.Linear(embedding_dim, 1))

        self.attention_linear = nn.ModuleDict()
        for i in range(0, len(self.feature_keys)):
            self.attention_linear[str(i)] = nn.Linear(embedding_dim * 4, 1)

        self.fusion_based2 = nn.ModuleDict()
        for i in range(0,len(self.feature_keys)):
            self.fusion_based2[str(i)] = nn.Linear(embedding_dim, 1)
        self.recommenders = nn.ModuleDict()
        for feature in feature_keys:
            self.recommenders[feature] = PresRecBlock(dataset, feature_keys, label_key, mode, self.feat_tokenizers, self.embeddings, self.linear_layers, self.label_tokenizer, embedding_dim, max_n_codes, device, n_time_step, n_heads, negative_ratio, tmpr, cl, feature).to(self.device)
    def generate_ddi_adj(self) -> torch.FloatTensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((self.label_size, self.label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj
    def get_ddi_loss(self, y_prob):
        mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
        ddi_loss = (mul_pred_prob * self.ddi_adj).sum() / (self.ddi_adj.shape[0] ** 2)
        return ddi_loss
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        embed = []
        intent_embed = []
        #self.iter += 1
        cl = torch.zeros(1, requires_grad=True)
        #ce = torch.zeros(1, requires_grad=True)
        for key in self.feature_keys:
            v_embed, cl1, ce, intents = self.recommenders[key](self.interpretability, self.iter, **kwargs)
            cl = cl1.clone().to(self.device) + cl.to(self.device)
            #ce = ce.clone().to(self.device) + ce
            embed.append(v_embed)
            intent_embed.append(intents)
        #predictions = self.output_layer_class(visit_embedding)
        all_embeds = torch.stack(embed, dim=1)#.permute(0,2,1)

        all_embeds = all_embeds#/normed_tensor
        all_embeds = all_embeds.flatten(start_dim = 1)

        predictions = self.output_layer_class(all_embeds)

        logits = predictions #self.activation(predictions)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)

        loss_ = self.get_loss_function()(logits,
                                             y_true)
        #ranking_loss_ = self.loss(logits, y_true)
        n = np.ones(len(y_true)) + 1
        y_prob = self.prepare_y_prob(logits)
        # __________________________mlml loss ______________________
        labels_index = self.label_tokenizer.batch_encode_2d(
            kwargs[self.label_key], padding=False, truncation=False
        )
        labels = batch_to_multihot(labels_index, self.label_size)
        index_labels = -np.ones((len(labels), self.label_size), dtype=np.int64)
        for idx, cont in enumerate(labels_index):
            cont = list(set(cont))
            index_labels[idx, : len(cont)] = cont
        index_labels = torch.from_numpy(index_labels)

        index_labels = index_labels.to(self.device)
        loss_mlml = self.mlml(y_prob, index_labels)
        # __________________________mlml loss ______________________
        ddi_loss = self.get_ddi_loss(y_prob)
        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.3] = 1
        y_pred[y_pred < 0.3] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]
        cur_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())
        ddi_treshhold = 0.04
        if cur_ddi_rate > ddi_treshhold:
            ddi_coef = 0.05
        else:
            ddi_coef = 0.0
        if self.cl > 0.001:
            if self.iter < 69 and self.cl > 0.001:
                #self.cl = 0
                #loss = loss_ + (ddi_coef*ddi_loss) + (0.05* loss_mlml)
                loss =  self.cl * (cl) / len(self.feature_keys) + loss_ + (0.05* loss_mlml) + ddi_coef*ddi_loss#loss_ + (ddi_coef*ddi_loss) + (0.05* loss_mlml) + self.cl * (cl) / len(self.feature_keys)

            #if self.cl == 0.0:
            #    loss = loss_ + (ddi_coef*ddi_loss) + (0.05* loss_mlml)
            else:
                loss = loss_ + self.cl * (cl) / len(self.feature_keys) + (0.05* loss_mlml)+ ddi_coef*ddi_loss#+ self.cl * (cl) / len(self.feature_keys) #+  (0.05* loss_mlml) + (ddi_coef*ddi_loss)
        else:
            loss = loss_+ ddi_coef*ddi_loss + (0.05* loss_mlml)#+ (0.05 * loss_mlml) #+ (ddi_coef * ddi_loss) + (0.05 * loss_mlml)



        #print('BCE', self.get_loss_function()(logits, y_true))
        print('CL', self.cl * (cl))


        self.iter += 1
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}


class PresRecBlock(BaseModel):
    def __init__(self, dataset: SampleBaseDataset, feature_keys, label_key: str, mode: str, feat_tokenizers, embeddings, linear_layers, label_tokenizer,
                 embedding_dim=128, max_n_codes=49, device=1, n_time_step=4,
                 n_heads=4, negative_ratio=0.1, tmpr=0.8, cl=0.01, feature_key = ''):  # n_rx = maximum number of rx codes in each visit,
        super().__init__(dataset, feature_keys, label_key, mode)

        hidden_size = embedding_dim
        self.embedding_dim = embedding_dim
        self.cl = cl
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.iter = 0
        self.feat_tokenizers = feat_tokenizers
        self.embeddings = embeddings
        self.linear_layers = linear_layers
        self.label_tokenizer = label_tokenizer
        self.embedding_dim = embedding_dim
        self.dropout_layer = nn.Dropout(p=0.01)

        #self.dual = DualPathRNN(input_size=embedding_dim, hidden_size = embedding_dim, output_size=embedding_dim)
        print('self_device', self.device)
        self.feature_key = feature_key
        self.inner_transformer = ct.InnerSelfAttention(hidden_size, n_heads, max_n_codes, f'cuda:{device}').to(self.device)
        self.inter_transformer = ct.InterSelfAttention(hidden_size, n_heads, max_n_codes, f'cuda:{device}').to(self.device)
        self.output_layer = nn.Linear(max_n_codes * int(hidden_size / n_heads), hidden_size)
        self.pooling_layer = nn.Linear(int(hidden_size / n_heads), 1)
        #        self.output_layer_class = nn.Linear(embedding_dim * n_time_step,
         #                                   self.get_output_size(self.get_label_tokenizer()))
        self.hidden_size = hidden_size
        self.n_time_step = n_time_step
        self.negative_ratio = negative_ratio
        self.n_heads = n_heads
        self.tmpr = tmpr
        self.dropout = nn.Dropout(0.2)
        self.value_inter_intra = nn.Linear(2* int(hidden_size / n_heads), int(hidden_size / n_heads))
        self.head_contrastive = nn.Linear(int(hidden_size / n_heads) * n_time_step, int(hidden_size / n_heads))
        self.head_contrastive2 = nn.Linear(int(hidden_size / n_heads), int(hidden_size / n_heads))
        self.retain_visit = RETAINLayer(feature_size =  embedding_dim, device=f'cuda:{device}')#ArkiLinearRetain(feature_size=embedding_dim)
        self.retain_intent = RETAINLayer(feature_size = embedding_dim, device=f'cuda:{device}')#ArkiLinearRetain(feature_size=embedding_dim)

        self.head_supervised_contrastive = nn.Linear(int(hidden_size / n_heads), self.n_heads)
        self.intent_embedding = nn.ModuleList()

        for _ in range(n_heads):
            self.intent_embedding.append(nn.Linear(hidden_size, int(hidden_size / n_heads)))
        #self.intent_concatenation = nn.Linear(hidden_size, int(hidden_size / n_heads))
        self.intent_classification = nn.Linear(int(hidden_size / n_heads) * n_time_step, n_heads)
        self.score_calculator_tensor = nn.Linear(2 * hidden_size, 1)
        self.temp = nn.Linear(max_n_codes * int(hidden_size / n_heads) * n_time_step,
                              self.get_output_size(self.get_label_tokenizer()))
        self.output_layer_class = nn.Linear(hidden_size * n_time_step,
                                            self.get_output_size(self.get_label_tokenizer()))
        self.values_concat = nn.Linear(2 * int(hidden_size / n_heads), int(hidden_size / n_heads))

        self.loss = nn.BCELoss()
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(max_n_codes * int(hidden_size / n_heads), 1),
            torch.nn.Softmax(dim=1)  # Apply softmax along dimension 1 (dim 1)
        )
        #self.nodes_concatenator =  nn.Linear(self.get_output_size(self.feat_tokenizers[self.feature_key])*int(hidden_size / n_heads), int(hidden_size / n_heads))
        self.nodes_concatenator =  nn.Linear(100*int(hidden_size / n_heads), int(hidden_size / n_heads))
        self.class_embed = nn.Linear(embedding_dim * 2,
                                        embedding_dim)
        self.visits_embed = nn.Linear(embedding_dim * n_time_step,
                                     embedding_dim)
        self.intent_embed = nn.Linear(embedding_dim * n_time_step,
                                        embedding_dim)
    def get_items_embedding(self, list_of_items):
        all_items = torch.Tensor(list_of_items).type(torch.int).to(self.device)
        all_items_embedding = self.embedding(all_items)
        return all_items_embedding

    def set_time_step(self, n_time):
        self.n_time_step = n_time

    def score_calculator(self, item, visit):

        score = (item * visit).sum(dim=1)
        return score

    def calculate_loss(self, positive_items, user, visit, constrastive_loss, epoch=None, contrastive_alpha=0.1):
        '''normal_pos_sample, normal_neg_sample = self.normal_sampling(positive_items)
        medical_pos_sample, medical_neg_sample = self.medical_sampling(positive_items)
        l_normal = self.BPR_Loss(normal_pos_sample, normal_neg_sample, user, visit)
        l_medical = self.BPR_Loss(medical_pos_sample, medical_neg_sample, user, visit)
        return (self.negative_ratio*l_medical) + ((1-self.negative_ratio)* l_normcontrastial)'''
        categories = np.zeros((len(visit), self.vocab_size - 1)).astype(float)
        for i in range(0, len(positive_items)):
            categories[i, positive_items[i]] = 1.0
        categories = torch.tensor(categories).type(torch.float)
        actual_loss = self.loss(visit.type(torch.float).to(self.device), categories.to(self.device))
        medical_loss = self.ontology_loss(positive_items, visit)
        print('binary cross entropy: ', ((1 - self.negative_ratio) * actual_loss) + self.negative_ratio * medical_loss)
        print('contrastive normalized: ', (contrastive_alpha * constrastive_loss))
        return ((
                            1 - self.negative_ratio) * actual_loss) + self.negative_ratio * medical_loss + contrastive_alpha * constrastive_loss


    def head_contrastive_loss_individual_head_anchor(self, attentioninners, intents, mask, type):  # REALS

        loss = torch.zeros(1, requires_grad=True)
        masks_time_inner = mask.type(torch.int).to(self.device)
        if type == 'inter':
            time_intent_ = intents * masks_time_inner.unsqueeze(-1).unsqueeze(-1)

            mask_clone = masks_time_inner.clone()
            for i in range(1, self.n_time_step):
                mask_clone[:, i] = mask[:, i].type(torch.int).to(self.device) * mask[:, i - 1].type(torch.int).to(
                    self.device)
            masks_time_inner = mask_clone.clone()
        else:
            time_intent_ = intents * masks_time_inner.unsqueeze(-1).unsqueeze(-1)

        attentioninner = attentioninners.to(self.device) * masks_time_inner.unsqueeze(-1).unsqueeze(-1).to(self.device)

        attentioninner = attentioninner.permute(0, 2, 1, 3).flatten(start_dim=2)
        time_intent_ = time_intent_.permute(0, 2, 1, 3).flatten(start_dim=2)
        inner = self.head_contrastive(attentioninner)
        time_intent = self.head_contrastive(time_intent_)
        # inter = self.head_contrastive(inter_t)
        inner_sim = torch.cosine_similarity(time_intent, inner, dim=-1)
        #pos = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
        #neg_loss = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
        for h in range(0, self.n_heads):
            #pos = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
            neg_loss = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
            pos = torch.exp(inner_sim[:, h] / self.tmpr)  # + torch.exp(inter_sim[:, h] / self.tmpr)
            for h_ in range(0, self.n_heads):
                if not h == h_:
                    neg_loss = neg_loss.clone().to(self.device) + torch.exp(
                        torch.cosine_similarity(time_intent[:, h, :], inner[:, h_, :], dim=-1) / self.tmpr)
                    neg_loss = neg_loss.clone().to(self.device) + torch.exp(
                        torch.cosine_similarity(inner[:, h, :], inner[:, h_, :], dim=-1) / self.tmpr)
                    #neg_loss = neg_loss.clone().to(self.device) + torch.exp(
                    #    torch.cosine_similarity(inner[:, h, :], inner[:, h_, :], dim=-1) / self.tmpr)
            loss = loss.clone().to(self.device) + ((-1 * torch.log(pos / (neg_loss + pos))).sum())
        return loss / attentioninner.size()[0]

    def head_contrastive_loss_individual(self, attentioninners, intents, mask, type):  # REALS

        loss = torch.zeros(1, requires_grad=True)
        masks_time_inner = mask.type(torch.int).to(self.device)
        if type == 'inter':
            time_intent_ = intents * masks_time_inner.unsqueeze(-1).unsqueeze(-1)

            mask_clone = masks_time_inner.clone()
            for i in range(1, self.n_time_step):
                mask_clone[:, i] = mask[:, i].type(torch.int).to(self.device) * mask[:, i - 1].type(torch.int).to(
                    self.device)
            masks_time_inner = mask_clone.clone()
        else:
            time_intent_ = intents * masks_time_inner.unsqueeze(-1).unsqueeze(-1)

        attentioninner = attentioninners.to(self.device) * masks_time_inner.unsqueeze(-1).unsqueeze(-1).to(self.device)

        attentioninner = attentioninner.permute(0, 2, 1, 3).flatten(start_dim=2)
        time_intent_ = time_intent_.permute(0, 2, 1, 3).flatten(start_dim=2)
        inner = self.head_contrastive(attentioninner)
        time_intent = self.head_contrastive(time_intent_)
        # inter = self.head_contrastive(inter_t)
        inner_sim = torch.cosine_similarity(time_intent, inner, dim=-1)
        #pos = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
        #neg_loss = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
        for h in range(0, self.n_heads):
            #pos = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
            neg_loss = torch.zeros((time_intent.size()[0]), requires_grad=True).to(self.device)
            pos = torch.exp(inner_sim[:, h] / self.tmpr)  # + torch.exp(inter_sim[:, h] / self.tmpr)
            for h_ in range(0, self.n_heads):
                if not h == h_:
                    neg_loss = neg_loss.clone().to(self.device) + torch.exp(
                        torch.cosine_similarity(time_intent[:, h, :], inner[:, h_, :], dim=-1) / self.tmpr)
                    neg_loss = neg_loss.clone().to(self.device) + torch.exp(
                        torch.cosine_similarity(time_intent[:, h, :], time_intent[:, h_, :], dim=-1) / self.tmpr)
                    #neg_loss = neg_loss.clone().to(self.device) + torch.exp(
                    #    torch.cosine_similarity(inner[:, h, :], inner[:, h_, :], dim=-1) / self.tmpr)
            loss = loss.clone().to(self.device) + ((-1 * torch.log(pos / (neg_loss + pos))).sum())
        return loss / attentioninner.size()[0]
    def head_contrastive_loss(self, attentioninner, attentioninter, intents, mask):

        loss1 = self.head_contrastive_loss_individual(attentioninner, intents, mask, 'inner')
        loss2 = self.head_contrastive_loss_individual(attentioninter, intents, mask, 'inter')

        return loss1 + loss2


    def ontology_loss(self, positive_items, visit):
        rx_atc, atc_rx = do.get_atc_onthology()
        categories = np.zeros((len(visit), self.vocab_size - 1)).astype(float)
        for i in range(0, len(positive_items)):
            for item in positive_items[i]:
                categories[i, atc_rx[rx_atc[item]]] = 1.0
        categories = torch.tensor(categories).type(torch.float)
        return self.loss(visit.type(torch.float).to(self.device), categories.to(self.device))

    def BPR_Loss(self, positive_sample, negative_sample, user, visit):
        positive_value = self.score_calculator(positive_sample, visit)
        negative_value = self.score_calculator(negative_sample, visit)
        loss = -1 * F.logsigmoid(positive_value - negative_value).sum()
        return loss
    def intent_level_classification_loss(self, intent, intent_labels):
        intents = torch.stack(intent, dim=0).flatten(start_dim=0, end_dim = 1)
        loss = nn.CrossEntropyLoss()
        intent_pred = self.intent_classification(intents.flatten(start_dim = 1))
        intent_labels_flat = intent_labels.flatten()
        intent_labels_hot = torch.nn.functional.one_hot(intent_labels_flat.type(torch.int64), self.n_heads)
        num = intent_labels_hot.size()[0]
        indices = torch.randperm(num)

        return loss(torch.sigmoid(intent_pred)[indices, :], intent_labels_hot.to(self.device)[indices, :].to(torch.float))
    def normal_sampling(self, positive_samples):  # Normal sampling with uniform
        positive_embedding = torch.Tensor(np.zeros((len(positive_samples), self.hidden_size))).to(self.device)
        negative_embedding = torch.Tensor(np.zeros((len(positive_samples), self.hidden_size))).to(self.device)
        for index, pos_samples in enumerate(positive_samples):
            positive_sample = torch.Tensor([np.random.choice(pos_samples)]).type(torch.int).to(self.device)
            positive_sample_embedding = self.embedding(positive_sample)
            negative_sample = []
            while len(negative_sample) == 0:
                negative_sample_c = random.randint(1, self.vocab_size - 1)
                if not negative_sample_c in pos_samples:
                    negative_sample.append(negative_sample_c)
            negative_sample = torch.Tensor(negative_sample).type(torch.int).to(self.device)
            negative_sample_embedding = self.embedding(negative_sample)
            positive_embedding[index, :] = positive_sample_embedding[0, :]
            negative_embedding[index, :] = negative_sample_embedding[0, :]
        return positive_embedding, negative_embedding

    def medical_sampling(self, positive_samples):  # Onthology-aware sampling
        positive_embedding = torch.Tensor(np.zeros((len(positive_samples), self.hidden_size))).to(self.device)
        negative_embedding = torch.Tensor(np.zeros((len(positive_samples), self.hidden_size))).to(self.device)
        rx_atc, atc_rx = do.get_rx_onthology()
        for index, pos_samples in enumerate(positive_samples):
            pos_sample = np.random.choice(pos_samples)
            positive_sample = torch.Tensor([pos_sample]).type(torch.int).to(self.device)
            positive_sample_embedding = self.embedding(positive_sample)
            negative_sample = []
            while len(negative_sample) == 0:
                if (rx_atc[pos_sample] == 0) or len(atc_rx[rx_atc[pos_sample]]) == 1:
                    negative_sample_c = random.randint(1, self.vocab_size - 1)
                    if not negative_sample_c in pos_samples:
                        negative_sample.append(negative_sample_c)
                else:
                    negative_sample_c = np.random.choice(atc_rx[rx_atc[pos_sample]])
                    if not positive_sample == negative_sample_c:
                        negative_sample.append(negative_sample_c)
            negative_sample = torch.Tensor(negative_sample).type(torch.int).to(self.device)
            negative_sample_embedding = self.embedding(negative_sample)
            positive_embedding[index, :] = positive_sample_embedding[0, :]
            negative_embedding[index, :] = negative_sample_embedding[0, :]
        return positive_embedding, negative_embedding

    def forward(self, interpretability, iter, **kwargs):
        input_info = self.dataset.input_info[self.feature_key]
        assert input_info["dim"] == 3 and input_info["type"] == str


        x = self.feat_tokenizers[self.feature_key].batch_encode_3d(kwargs[self.feature_key])
        x_val = torch.tensor(x, dtype=torch.long, device=self.device)
        diff_time = self.n_time_step - x_val.size()[1]
        if self.n_time_step - x_val.size()[1] > 0:
            zeros = torch.zeros(x_val.size()[0], self.n_time_step - x_val.size()[1], x_val.size()[2]).type(
                torch.LongTensor).to(self.device)
            x_val = torch.cat([x_val, zeros], dim=1)

        x = self.embeddings[self.feature_key](x_val)

        mask = torch.sum(torch.sum(x, dim=2) != 0, dim=2) != 0

        intents = []
        intent_labels = []


        flatten_x = x.sum(dim=2)
        for i in range(0, self.n_heads):
            intent_labels.append(i * torch.ones((flatten_x.size()[0])))
            intents.append(self.intent_embedding[i](flatten_x))
        intents_representation = torch.stack(intents,dim = 3).flatten(start_dim = 2)


        intents = torch.stack(intents, dim=2)
        embedding = x
        max_code = x_val.size()[-1]
        for t in range(0, self.n_time_step):
            at_inn, q, k, v, att_score = self.inner_transformer(embedding[:, t, :, :], x_val[:, t, :],
                                                                max_code, interpretability, iter)  # [Batch_size x Num_of_heads x Seq_length]
            if t == 0:
                # attention_inn = torch.unsqueeze(at_inn, 1)
                queries_t = torch.unsqueeze(q, 1)
                keys_t = torch.unsqueeze(k, 1)
                values_t = torch.unsqueeze(v, 1)
                values_inter_t = torch.unsqueeze(v, 1)
                values_inter_t_2 = torch.unsqueeze(v, 1)

            else:
                # attention_inn = torch.cat((attention_inn, torch.unsqueeze(at_inn, 1)), 1)
                queries_t = torch.cat((queries_t, torch.unsqueeze(q, 1)), 1)
                keys_t = torch.cat((keys_t, torch.unsqueeze(k, 1)), 1)
                values_t = torch.cat((values_t, torch.unsqueeze(v, 1)), 1)
                # att_score_inn.append(att_score)

        for t in range(0, self.n_time_step):
            if t == 1:
                at_int, att_score, values, contex2 = self.inter_transformer(x_val[:,
                                                                   t - 1, :], x_val[:,
                                                                              t, :],
                                                                   queries_t[:, t - 1, :, :],
                                                                   keys_t[:, t, :, :], embedding[:, t, :, :],
                                                                   embedding[:, t - 1, :, :],
                                                                   values_inter_t[:, -1, :, :], values_t[:, t, :, :],
                                                                   max_code, interpretability, iter)
                values_inter_t = torch.cat((values_inter_t, torch.unsqueeze(values, 1)), 1)
                values_inter_t_2 = torch.cat((values_inter_t_2, torch.unsqueeze(contex2, 1)), 1)

            if t > 1:
                at_int, att_score, values, contex2 = self.inter_transformer(x_val[:,
                                                                   t - 1, :], x_val[:,
                                                                              t, :],
                                                                   queries_t[:, t - 1, :, :],
                                                                   keys_t[:, t, :, :], embedding[:, t, :, :], embedding[:, t-1, :, :],values_inter_t[:, -1, :, :],values_t[:, t, :, :],
                                                                   max_code,interpretability, iter)  # [Batch_size x Num_of_heads x Seq_length]
                values_inter_t = torch.cat((values_inter_t, torch.unsqueeze(values, 1)), 1)
                values_inter_t_2 = torch.cat((values_inter_t_2, torch.unsqueeze(contex2, 1)), 1)

        contrastive_loss = self.head_contrastive_loss(values_t.sum(dim=3), values_inter_t_2.sum(dim=3), intents, mask)
        output_t = values_inter_t#values_t + values_inter_t


        output_t = output_t.sum(3).flatten(start_dim = 2)

        output_t_intent = intents_representation  # .sum(3).flatten(start_dim=2)

        output_t = self.dropout_layer(output_t)
        output_t_intent = self.dropout_layer(output_t_intent)

        visit_embed = self.retain_visit(output_t, mask, iter,interpretability, "visit")#self.retain_visit(output_t, output_t_intent, mask)
        output_t_intent = self.retain_visit(output_t_intent,mask, iter, interpretability, "intent")#self.retain_visit(output_t_intent, output_t, mask)

        #visit_embed = torch.flatten(output_t,start_dim=2)
        #mask = mask.type(torch.int).unsqueeze(-1).expand_as(visit_embed)  # Broadcasting to match input_data shape

        # Apply the mask to the input_data
        masked_input = visit_embed #* mask
        masked_intent = output_t_intent #* mask



        visit_embed = self.class_embed(torch.cat([masked_input.to(self.device), masked_intent.to(self.device)], dim=-1))

        return visit_embed.to(self.device), contrastive_loss.to(self.device), 0, visit_embed.to(self.device)#intents_representation#, intents_representation

