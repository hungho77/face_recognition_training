from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.loss = "Arcface"
config.output = "./trained_models/arcface_finetune_tinyface_r100_v2"
config.network = "r100"
config.resume = False
config.pretrained_backbone_model = "./weights/WF12M_IResNet100_AdaFace_CFSM_model.pt"
config.pretrained_partial_fc_path = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 20
config.dali = False
config.frequent = 10
config.score = None

config.rec_train = "/home/hunght21/data/tinyface/Training_Set"
config.num_classes = 2570
config.num_image = 7804
config.num_epoch = 25
config.warmup_epoch = 2
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_targets = None
config.save_all_states = True


config.rs_ratio = 1
config.pretrained_synthesis_model = None
config.epsilon = 0.314 
config.alpha = 0.314     
config.k = 1