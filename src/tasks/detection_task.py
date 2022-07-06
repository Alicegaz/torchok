from pydantic import BaseModel
from collections import defaultdict
import torch

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, DETECTION_NECKS, DETECTION_HEADS, DETECTION_HATS
from .base_task import BaseTask


class DetectionParams(BaseModel):
    num_backbone_outs: int
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    backbone_name: str
    backbone_params: dict = {}
    neck_name: str
    neck_params: dict = {}
    head_name: str
    head_params: dict = {}
    hat_name: str
    hat_params: dict = {}
    freeze_backbone: bool = False
    skip_loss_on_eval: bool = True
    bbx_conf_thr: float = 0.5


@TASKS.register_class
class DetectionTask(BaseTask):
    config_parser = DetectionParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.example_input_array = None
        self.bbx_conf_thr = self.params.bbx_conf_thr
        self.num_backbone_outs = self.params.num_backbone_outs

        # create backbone
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params) #TODO: frozen_batch_norm

        # create neck
        self.neck = DETECTION_NECKS.get(self.params.neck_name)(**self.params.neck_params)

        # create head
        self.head = DETECTION_HEADS.get(self.params.head_name)(**self.params.head_params)

        # create hat
        self.hat = DETECTION_HATS.get(self.params.hat_name)(**self.params.hat_params)
        #TODO: transforms https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/datasets/transforms.py
        #TODO: prepare loss inputs, prepare loss outputs, hat
        #TODO: dilation in resnet backbone
        
    def forward(self, x, img_metas):
        # with torch.set_grad_enabled(not self.params.freeze_backbone and self.training):
        _, features = self.backbone.forward_backbone_features(x)
        # start_level = len(features) - self.num_backbone_outs + 1
        # print(start_level)
        features = features[-self.num_backbone_outs:]
        # print("after crop", features)
        hs, init_reference, inter_reference = self.neck(features, img_metas)
        output = self.head(hs, init_reference, inter_reference)
        return output

    def configure_optimizers(self):
        modules = [self.neck, self.head]
        if not self.params.freeze_backbone:
            modules.append(self.backbone)
        optimizer = create_optimizer(modules, self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def _parse_batch(self, batch):
        input_data = batch['input']
        masks = batch['mask']
        n_images = len(input_data)
        gt_bboxes = [batch['target_bboxes'][i][:batch['bbox_count'][i]] for i in range(n_images)]
        gt_bboxes_orig = [batch['target_bboxes_orig'][i][:batch['bbox_count'][i]] for i in range(n_images)]
        gt_labels = [batch['target_classes'][i][:batch['bbox_count'][i]] for i in range(n_images)]
        img_metas = [{
            'pad_shape': batch['pad_shape'][i],
            'img_shape': batch['img_shape'][i]
        } for i in range(n_images)]

        return input_data, gt_bboxes, gt_labels, img_metas, masks, gt_bboxes_orig

    def forward_train(self, batch):
        input_data, gt_bboxes, gt_labels, img_metas, masks, gt_bboxes_orig = self._parse_batch(batch)
        output = self.forward(input_data, masks)
        # output = self.prepare_loss_inputs(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas)
        targets = []
        for i in range(len(gt_labels)):
            targets.append({'target_classes':gt_labels[i],
            "target_bboxes": gt_bboxes[i]})
        return output, targets

    def _criterion_forward(self, output, targets):
        loss_dict = self.criterion(**{"output": output,
                                      "target": targets})
        # print(loss_dict)
        weight_dict = self.criterion.losses[0].weight_dict
        loss_keys = list(loss_dict.keys() - {'class_error', 'cardinality_error'})
        loss = weight_dict[loss_keys[0]] * loss_dict[loss_keys[0]]
        for k in loss_keys[1:]:
            loss += loss_dict[k]*weight_dict[k]
        # TODO: logging
        return loss, loss_dict['class_error']

    def training_step(self, batch, batch_idx):
        output, targets = self.forward_train(batch)
        loss, class_error = self._criterion_forward(output, targets)
        # self.metric_manager.update('train', **output)
        self.log('train/class_error', class_error.item(), on_step=True, on_epoch=False)
        return loss

    def forward_eval(self, batch):
        input_data, gt_bboxes, gt_labels, img_metas, masks, gt_bboxes_orig = self._parse_batch(batch)
        orig_img_sizes = torch.stack([img_m['img_shape'] for img_m in img_metas]) #TODO: was ["img_shape"]
        output = self.forward(input_data, masks)
        results = self.hat(output, orig_img_sizes) #TODO: is number of masks different for each sample
        # cls_preds = []
        # bbox_preds = []
        preds_pairs = []
        for res in results: # [ {'boxes': [[], []], 'labels': [1, 1]}, {'boxes': [[], []], 'labels': [1, 1]}]
            # cls_preds.append(res['labels']) #TODO: stack or not stack based on what is expected by metrics manager
            # bbox_preds.append(res['boxes']) #TODO: stack or not stack, based on what is expected by metrics manager
            flt_pair = defaultdict(list)
            for res_i_b, res_i_l, res_i_s in zip(res['boxes'], res['labels'], res['scores']):
                # if res_i_s.detach().cpu().item() > self.bbx_conf_thr:
                # if res_i_l.detach().cpu().item() == 0:
                    # print("label 0", res_i_l, res_i_b, res_i_s)
                # print(res_i_s.detach().cpu().item())
                    flt_pair['boxes'].append(res_i_b)
                    flt_pair['labels'].append(res_i_l)
            print(flt_pair['boxes'][0])
            preds_pairs.append([torch.stack(flt_pair['boxes']), torch.stack(flt_pair['labels'])])
        gt_pairs = []
        for i in range(len(gt_labels)):
            gt_pairs.append([gt_bboxes_orig[i], gt_labels[i]])
        # targets = []
        # for i in range(len(gt_labels)):
        #     targets[i]['taget_classes'] = gt_labels[i]
        #     targets[i]["target_bboxes"] = gt_bboxes[i]

        # output = {
        #     'prediction': torch.stack([bbox_preds, cls_preds.unsqueeze(1)], dim=1), #TODO: unsqueeze why?
        #     'target': torch.stack([gt_bboxes, gt_labels.unsqueeze(1)], dim=1) #TODO: unsqueeze why?
        # }
        return preds_pairs, gt_pairs

    def _eval_step(self, batch, batch_idx, mode):
        output, targets = self.forward_eval(batch)
        if not self.params.skip_loss_on_eval:
            loss = self.criterion(**{"output": output,
                                      "target": targets})
            self.metric_manager.update(mode, prediction=output, target=targets) # TODO: pass targets

            return loss
        else:
            self.metric_manager.update(mode, prediction=output, target=targets) #TODO: pass targets

            return torch.tensor(0.)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'test')
