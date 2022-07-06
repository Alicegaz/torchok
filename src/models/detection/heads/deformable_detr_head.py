import torch
from torch import nn
from torch.nn import functional as F
from src.registry import HEADS, DETECTION_HEADS

import math
from ..necks.transformer import inverse_sigmoid

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_class
@DETECTION_HEADS.register_class
class DeformableDetrHead(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, num_pred, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.class_embed = nn.Linear(input_dim, num_classes)
        self.bbox_embed = MLP(input_dim, input_dim, 4, 3)
        self._init_parameters()
        # num_pred = transformer.decoder.num_layers #TODO: initialize in task

        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        # self.transformer.decoder.bbox_embed = None

    def _init_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

    def forward(self, hs, init_reference, inter_reference):
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_reference[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            # outputs_class = outputs_class.sigmoid()
            # outputs_coord = tmp.relu()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} #TODO: format output
        # return outputs_class[-1], outputs_coord[-1]
        return out
