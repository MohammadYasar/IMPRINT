import torch
import torch.nn as nn
import torch.nn.functional as F

class KeylessAttention(nn.Module):
    def __init__(self, feature_embed_size):
        super(KeylessAttention, self).__init__()
        self.feature_embed_size = feature_embed_size
        self.attention_module = nn.Conv1d(self.feature_embed_size, 1, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.permute(1,2,0) # T,B,C -> B,C,T
        # print ("x {}".format(x.shape))
        weights = self.softmax(self.attention_module(x))# .unsqueeze(-1)
        # print ("weights {} x {}".format(weights.shape, x.shape))
        weights_unexpanded = weights
        weights = weights.expand_as(x)
        outputs = x*weights
        # print ("outputs ",outputs.shape, weights)
        outputs = outputs.permute(2,0,1) # B,C,T -> T,B,C
        outputs = torch.sum(outputs, dim=0).unsqueeze(0)
        return outputs, weights_unexpanded
