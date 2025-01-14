from easydict import EasyDict as edict

config = edict()
config.loss = "CombinedMarginLoss"
config.output = "./trained_models/adaface_tinyface_r100"
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

config.rec_train = "../cfp-dataset/Data/Images/"
config.rec_val = "../cfp-dataset/Data/Val/"
config.num_classes = 450
config.num_image = 6302
config.num_epoch = 100
config.warmup_epoch = 2
# config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.val_targets = ["cfp_fp"]
config.save_all_states = True


config.rs_ratio = 0.75
config.pretrained_synthesis_model = None
config.epsilon = 0.314   
config.alpha = 0.314     
config.k = 1