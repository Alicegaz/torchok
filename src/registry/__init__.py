from .registry import Registry

CALLBACKS = Registry('callbacks')
TASKS = Registry('tasks')
DATASETS = Registry('datasets')
OPTIMIZERS = Registry('optimizers')
SCHEDULERS = Registry('schedulers')
METRICS = Registry('metric')
LOSSES = Registry('losses')
POOLINGS = Registry('pooling_heads')
HEADS = Registry('heads')
CLASSIFICATION_HEADS = Registry('classification_heads')
SEGMENTATION_HEADS = Registry('segmentation_heads')
SEGMENTATION_MODELS = Registry('segmentation_models')
DETECTION_HEADS = Registry('detection_heads')
DETECTION_HATS = Registry('detection_hats')
DETECTION_NECKS = Registry('detection_necks')
