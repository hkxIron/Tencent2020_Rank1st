import argparse
import torch
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 将bert与 transformer模型融合
class ClassifyModel(nn.Module):
    def __init__(self, args):
        super(ClassifyModel, self).__init__()
        args.out_size = len(args.dense_features)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # 创建BERT模型，并且导入预训练模型
        config = RobertaConfig.from_pretrained(args.pretrained_model_path)
        config.output_hidden_states = True
        args.hidden_size = config.hidden_size
        args.num_hidden_layers = config.num_hidden_layers
        self.bert_text_layer = RobertaModel.from_pretrained(args.pretrained_model_path, config=config)
        self.text_linear = nn.Linear(in_features=args.text_dim + args.vocab_dim_v1 * len(args.text_features),
                                     out_features=args.hidden_size)
        logger.info("Load linear from %s", os.path.join(args.pretrained_model_path, "linear.bin"))
        self.text_linear.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, "linear.bin")))
        logger.info("Load embeddings from %s", os.path.join(args.pretrained_model_path, "embeddings.bin"))

        self.text_embeddings = nn.Embedding.from_pretrained(
            torch.load(os.path.join(args.pretrained_model_path, "embeddings.bin"))['weight'],
            freeze=True)
        args.out_size += args.hidden_size * 2

        # 创建fusion-layer模型，随机初始化
        config = RobertaConfig()
        config.num_hidden_layers = 4
        config.intermediate_size = 2048
        config.hidden_size = 512
        config.num_attention_heads = 16
        config.vocab_size = 5
        self.fusion_text_layer = RobertaModel(config=config)
        self.fusion_text_layer.apply(self._init_weights)
        self.text_linear_1 = nn.Linear(args.text_dim_1 + args.hidden_size, 512)
        self.text_linear_1.apply(self._init_weights)
        self.norm = nn.BatchNorm1d(args.text_dim_1 + args.hidden_size)
        args.out_size += 1024

        # 创建分类器，随机初始化
        self.classifierHead = ClassificationHead(args)
        self.classifierHead.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self,
                dense_features,
                text_features,
                text_ids,
                text_masks,
                fusion_text_features,
                fusion_text_masks,
                labels=None):

        outputs = []
        # 获取浮点数，作为分类器的输入
        outputs.append(dense_features.float())

        # 获取BERT模型的hidden state，并且做max pooling和mean pooling作为分类器的输入
        text_masks = text_masks.float()
        text_embedding = self.text_embeddings(text_ids).view(text_ids.size(0), text_ids.size(1), -1)  # reshape
        text_features = torch.cat((text_features.float(), text_embedding), -1)  # concat
        text_features = torch.relu(self.text_linear(self.dropout(text_features)))  # relu
        hidden_states = self.bert_text_layer(inputs_embeds=text_features, attention_mask=text_masks)[0]  # bert_text_layer

        embed_mean = (hidden_states * text_masks.unsqueeze(-1)).sum(1) / text_masks.sum(1).unsqueeze(-1)
        embed_mean = embed_mean.float()
        embed_max = hidden_states + (1 - text_masks).unsqueeze(-1) * (-1e10)
        embed_max = embed_max.max(1)[0].float()
        # bert的embedding的mean, max作为分类器的输入
        outputs.append(embed_mean)
        outputs.append(embed_max)

        # 获取fusion-layer的hidden state，并且做max pooling和mean pooling作为分类器的输入
        fusion_text_masks = fusion_text_masks.float()
        fusion_text_features = torch.cat((fusion_text_features.float(), hidden_states), -1)
        batch, seq_length, embedding_dim = fusion_text_features.size()
        fusion_text_features = self.norm(fusion_text_features.view(-1, embedding_dim))\
            .view(batch, seq_length, embedding_dim)
        fusion_text_features = torch.relu(self.text_linear_1(fusion_text_features))
        hidden_states = self.fusion_text_layer(inputs_embeds=fusion_text_features,
                                               attention_mask=fusion_text_masks)[0]  # transfromer fusion
        embed_mean = (hidden_states * fusion_text_masks.unsqueeze(-1)).sum(1) / fusion_text_masks.sum(1).unsqueeze(-1)
        embed_mean = embed_mean.float()
        embed_max = hidden_states + (1 - fusion_text_masks).unsqueeze(-1) * (-1e10)
        embed_max = embed_max.max(1)[0].float()
        outputs.append(embed_mean)
        outputs.append(embed_max)

        # 将特征(bert max/mean pooling+fusion layer)输入分类器，得到20分类的logits
        # 年龄10维,性别2维,交叉之后就是20维
        final_hidden_state = torch.cat(outputs, dim=-1)
        logits = self.classifierHead(final_hidden_state)

        # 返回loss或概率结果
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            # prob:[batch, age, gender]
            prob = torch.softmax(logits, -1)
            # age_probs:[batch, age], 将每个age下的各个gender相加就可以得到该age的概率
            age_probs = prob.view(-1, 10, 2).sum(dim=2,keepdims=False)
            # gender_probs:[batch, gender]
            gender_probs = prob.view(-1, 10, 2).sum(1)
            return age_probs, gender_probs

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.norm = nn.BatchNorm1d(args.out_size)
        self.dense = nn.Linear(in_features=args.out_size, out_features=args.linear_layer_size[0])
        self.batch_norm_1 = nn.BatchNorm1d(args.linear_layer_size[0])
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.dense_1 = nn.Linear(in_features=args.linear_layer_size[0], out_features=args.linear_layer_size[1])
        self.batch_norm_2 = nn.BatchNorm1d(args.linear_layer_size[1])
        # out_proj:[batch, num_label=20]
        self.out_proj = nn.Linear(in_features=args.linear_layer_size[1], out_features=args.num_label) # 20维

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.batch_norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.batch_norm_2(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
