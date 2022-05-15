class Configuration():
    def __init__(self):
        self.PROJECT_NAME = 'pytorch-project-template'
        self.LOG_DIR = "./log/" #log dir and saved model dir
        self.DATA_DIR = "/dataset/"
        self.DEVICE_ID = "0"

        #data loader
        self.DATALOADER_NUM_WORKERS = 4

        #basemodel
        self.INPUT_SIZE = [256, 256] #HxW
        self.MODEL_NAME = "resnet50"
        self.PRETRAIN_PATH = ""
        self.PRETRAIN_CHOICE = 'imagenet'

        #loss
        self.LOSS_TYPE = 'softmax'

        #solver
        self.BATCHSIZE = 64
        self.OPTIMIZER = 'Adam'
        self.BASE_LR = 0.001
        self.WEIGHT_DECAY = 0.0005
        self.MOMENTUM = 0.9

        self.STEPS = [30,50,70]
        self.GAMMA = 0.1
        self.WARMUP_FACTOR = 0.01
        self.WARMUP_EPOCHS = 10
        self.WARMUP_METHOD = "linear" #option: 'linear','constant'
        self.LOG_PERIOD = 50 #iteration of display training log
        self.CHECKPOINT_PERIOD = 5 #save model period
        self.EVAL_PERIOD = 5 #validation period
        self.MAX_EPOCHS = 90

        #test
        self.TEST_BATCH = 128
        self.TEST_WEIGHT = ''

