import torch
import torch.nn.functional as F
from torch import nn
from .transformer import DeformableTransformer
from src.models.layers.position_encoder import PositionEmbeddingSine, PositionEmbeddingLearned

from src.registry import DETECTION_NECKS


# def img_metas2mask(img_metas):
#     lens = [meta_i['img_shape'] for meta_i in img_metas]
#     len_mask = torch.zeros(bs, max_h, 1) + max_w
#     for i, len_i in enumerate(lens):
#         len_mask[i, torch.arange(len_i[0]), :] = len_i[1]
#     max_h, max_w = masks[0]['pad_shape'].shape[-2:] #TODO: are they all equal
#
#     mask = torch.arange(max_w)[None, :].unsqueeze(0) \
#            < len_mask  # lens[:, None].repeat_interleave(max_h, dim=-1).unsqueeze(-1)
#     return mask


@DETECTION_NECKS.register_class
class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone_outs_channels,
                 num_backbone_outs, hidden_dim,
                 num_queries, num_feature_levels,
                 transformer_params: dict = {},
                 pos_emb_name: str = 'sine'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        n_steps = hidden_dim // 2
        if pos_emb_name == "sine":
            self.position_encoder = PositionEmbeddingSine(n_steps, normalize=True)
        else:
            self.position_encoder = PositionEmbeddingLearned(n_steps)
        self.transformer = DeformableTransformer(**transformer_params)
        hidden_dim = self.transformer.d_model
        self.num_feature_levels = num_feature_levels
        kwargs = [{"kernel_size": 3, "stride": 2, "padding": 1}, {"kernel_size": 1}]
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # num_backbone_outs = len(backbone.strides)
        input_proj_list = []
        for i in range(num_feature_levels):
            if i > num_backbone_outs - 1 and num_feature_levels > 1:
                kwargs_i = kwargs[0]
            else:
                kwargs_i = kwargs[1]
                in_channels = backbone_outs_channels[i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, **kwargs_i),
                nn.GroupNorm(32, hidden_dim),
            ))
            if i > num_backbone_outs - 1:
                in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        self._init_parameters()
        self.transformer.decoder.bbox_embed = None

    def _init_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, features, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)
        # features, pos = self.backbone(samples)
        # TODO: add positional encodding
        srcs = []
        poses = []
        masks = []
        # TODO: how to read img_metas, is dict returned ok
        for l_i, feat in enumerate(features):  # check the order of features of different level
            mask_i = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            pos = self.position_encoder(feat, mask_i).to(feat.dtype)  # check dtype, should match dtype of feats
            src = self.input_proj[l_i](feat)
            srcs.append(src)
            poses.append(pos)
            masks.append(mask_i)


        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l_i in range(_len_srcs, self.num_feature_levels):
                if l_i == _len_srcs:
                    src = self.input_proj[l_i](features[-1])
                else:
                    src = self.input_proj[l_i](srcs[-1])
                mask_i = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.position_encoder(src, mask_i).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
        # print(poses[0])

        query_embeds = self.query_embed.weight

        hs, init_reference, inter_references = self.transformer(srcs, masks, poses, query_embeds)

        # outputs_classes = []
        # outputs_coords = []
        # for lvl in range(hs.shape[0]):
        #     if lvl == 0:
        #         reference = init_reference
        #     else:
        #         reference = inter_references[lvl - 1]
        #     reference = inverse_sigmoid(reference)
        #     outputs_class = self.class_embed[lvl](hs[lvl])
        #     tmp = self.bbox_embed[lvl](hs[lvl])
        #     if reference.shape[-1] == 4:
        #         tmp += reference
        #     else:
        #         assert reference.shape[-1] == 2
        #         tmp[..., :2] += reference
        #     outputs_coord = tmp.sigmoid()
        #     outputs_classes.append(outputs_class)
        #     outputs_coords.append(outputs_coord)
        # outputs_class = torch.stack(outputs_classes)
        # outputs_coord = torch.stack(outputs_coords)
        #
        # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return hs, init_reference, inter_references

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
