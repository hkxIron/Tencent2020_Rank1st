import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
# from transformers.modeling_bert import BertLayerNorm
# import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class BertModel(nn.Module):
    def __init__(self, bert_encoder, config, args):
        super(BertModel, self).__init__()
        self.bert_encoder = bert_encoder
        self.text_embeddings = nn.Embedding(num_embeddings=args.vocab_size_v1, embedding_dim=args.vocab_dim_v1)
        self.text_embeddings.apply(self._init_weights)  # 初始化权重
        self.text_linear = nn.Linear(in_features=args.text_dim + args.vocab_dim_v1 * len(args.text_features),
                                     out_features=config.hidden_size)
        self.text_linear.apply(self._init_weights)

        self.lm_head_layer_list = []
        for vocab_size in args.vocab_size:
            # 不同的feature,有不同的vocab_size,比如ad_id, product_id
            # 注意,不同层的输出vocab_size不一样,即后面会做softmax分类
            self.lm_head_layer_list.append(
                nn.Linear(in_features=config.hidden_size, out_features=vocab_size, bias=False))

        self.lm_head_layer_list = nn.ModuleList(self.lm_head_layer_list)
        self.config = config
        self.args = args

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inputs, inputs_ids, masks, labels):
        inputs_embedding = self.text_embeddings(inputs_ids) \
            .view(inputs.size(0), inputs.size(1), -1)  # view类似于tf.reshape

        inputs = torch.cat((inputs.float(), inputs_embedding), -1)  # concatnate(A, B, axis)
        inputs = torch.relu(self.text_linear(inputs))
        bert_encode_out = self.bert_encoder(inputs_embeds=inputs, attention_mask=masks.float())[0]

        loss = 0
        # 多个head的loss, 每个head代表不同的feature,有的是ad_id,有的是product_id,有的是advertiser_id
        # 比如 head_list: [ad_id, product_id, advertiser_id, ...]
        for idx, (feature_output_head_layer, vec_id_dim_istrain) in enumerate(
                zip(self.lm_head_layer_list, self.args.text_features)):
            is_train = vec_id_dim_istrain[3]
            if is_train:
                # labels:[batch, seq_length, feature_num]
                # 将非mask位置的bert output取出来
                outputs_tmp = bert_encode_out[labels[:, :, idx].ne(-100)]  # ne: not_equal
                # 线性映射layer前向传播
                prediction_scores = feature_output_head_layer(outputs_tmp)

                labels_tmp = labels[:, :, idx]
                # 将非mask位置的label取出来
                labels_tmp = labels_tmp[labels_tmp.ne(-100)].long()

                loss_fct = CrossEntropyLoss()
                # loss层前向传播
                masked_lm_loss = loss_fct(prediction_scores, labels_tmp)
                # 不同feature的mask lm loss相加
                loss += masked_lm_loss
        return loss
