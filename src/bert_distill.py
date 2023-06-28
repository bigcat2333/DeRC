from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, ModuleList
import torch.nn.functional as F
import torch
import logging

from clf_distill_loss_functions import ClfDistillLossFunction


class BertDistill(BertPreTrainedModel):
    """Pre-trained BERT model that uses our loss functions"""

    def __init__(self, config, num_labels, loss_fn: ClfDistillLossFunction, lower_loss_fn_type=None):
        super(BertDistill, self).__init__(config)
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        # if lower_loss_fn_type and lower_loss_fn_type=='plain':
        #     self.lower_loss_fn = ClfDistillLossFunctions.Plain()
        # elif lower_loss_fn_type and lower_loss_fn_type!='plain':
        #     print("lower_loss_fn_type not supported")
        #     exit()
        self.lower_loss_fn = None
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax()
        self.num_layers = 12
        self.debias_layer = 3
        self.classifier_list = ModuleList([nn.Linear(config.hidden_size, num_labels) for i in range(self.num_layers)])
        # self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.projection = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None, global_step=None, total_step=None, anti_bias_label=None):
        output_list, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        pooled_output_list = [self.bert.pooler(x) for x in output_list]
        logits_list = []
        for i in range(len(pooled_output_list)):
            if i == len(pooled_output_list)-1:
                if labels is not None:
                    # logits_list.append(self.classifier_list[i](self.dropout(pooled_output_list[i]) + self.projection(self.dropout(pooled_output_list[self.debias_layer]).detach())))
                    logits_list.append(self.classifier_list[i](self.dropout(pooled_output_list[i]) + self.dropout(pooled_output_list[self.debias_layer]).detach()))
                else:
                    logits_list.append(self.classifier_list[i](self.dropout(pooled_output_list[i])))
            else:
                logits_list.append(self.classifier_list[i](self.dropout(pooled_output_list[i])))

        logits = logits_list[-1]
        if labels is None:
            return logits_list[-1], logits_list[3]

        if anti_bias_label is None: 
            if self.lower_loss_fn:
                loss_lower = self.lower_loss_fn.forward(pooled_output_list[self.debias_layer], logits_list[self.debias_layer], bias, teacher_probs, labels)
            else:
                loss_lower = self.loss_fn.forward(pooled_output_list[self.debias_layer], logits_list[self.debias_layer], bias, teacher_probs, labels)
            loss_upper = self.loss_fn.forward(pooled_output_list[11], logits_list[11], bias, teacher_probs, labels)

        
            loss = 0.5 * loss_lower + 0.5 * loss_upper
            return logits, loss
        else:
            loss_lower = self.loss_fn.forward(pooled_output_list[self.debias_layer], logits_list[self.debias_layer], bias, teacher_probs, labels, False)
            loss_upper = self.loss_fn.forward(pooled_output_list[11], logits_list[11], bias, teacher_probs, labels, False)

            loss = 0.5 * loss_lower.mean() + 0.5 * loss_upper.mean()

            anti_bias_mask = (anti_bias_label == 1).int()
            bias_mask = (anti_bias_label == 0).int()
            loss_lower_anti_bias_loss = (loss_lower * anti_bias_mask).sum() / anti_bias_mask.sum()
            loss_lower_bias_loss = (loss_lower * bias_mask).sum() / bias_mask.sum()
            loss_upper_anti_bias_loss = (loss_upper * anti_bias_mask).sum() / anti_bias_mask.sum()
            loss_upper_bias_loss = (loss_upper * bias_mask).sum() / bias_mask.sum()

            return logits, loss, loss_lower_anti_bias_loss, loss_lower_bias_loss, loss_upper_anti_bias_loss, loss_upper_bias_loss

    def forward_and_log(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(self.dropout(pooled_output))
        if labels is None:
            return logits
        loss = self.loss_fn.forward(pooled_output, logits, bias, teacher_probs, labels)

        cel_fct = CrossEntropyLoss(reduction="none")
        indv_losses = cel_fct(logits, labels).detach()
        return logits, loss, indv_losses
    
    def forward_analyze(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, bias=None, teacher_probs=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(self.dropout(pooled_output))

        if labels is None and not self.training:
            return logits, pooled_output
        else:
            raise Exception("should be called during eval and "
                            "labels should be none")

