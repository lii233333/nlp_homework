import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication,self).__init__()
        self.model_name = 'hfl/chinese-bert-wwm'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768,15)

    def forward(self,x,device):
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                max_length=148, pad_to_max_length=True)      #tokenize、add special token、pad

        input_ids = torch.tensor(batch_tokenized['input_ids']).to(device)
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).to(device)

        hiden_outputs = self.model(input_ids,attention_mask=attention_mask)

        outputs = hiden_outputs[0][:,0,:]

        output = self.fc(outputs)

        return output








