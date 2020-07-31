from loguru import logger


class Config(object):
    def __init__(self):
        self.action = 'train'  # train or test
        logger.add('logs/{time: YYYY-MM-DD_HH-mm-ss}.log')
        self.logger = logger
        self.gpu_id = 0
        # path to the h5file containing the preprocessed dataset
        self.input_h5 = 'data/reidtalk.h5'
        # path to the json file containing additional info and vocab
        self.input_json = 'data/reidtalk.json'
        # path to CNN prototxt file in Caffe format.
        self.cnn_proto = 'model/VGG_ILSVRC_16_layers_deploy.prototxt'
        # path to VGG-16 Visual CNN
        self.cnn_model = 'model/VGG16_iter_50000.caffemodel'
        # path to a model checkpoint to initialize model weights from. Empty = don't
        self.start_from = ''
        self.neg_time = 3
        # the encoding size of each token in the vocabulary, and the image.
        self.input_encoding_size = 512
        # size of the rnn in number of hidden nodes in each layer
        self.rnn_size = 512
        self.batch_size = 32
        # clip gradients at this value
        # (note should be lower than usual 5 because we normalize grads by both batch and seq_length)
        self.grad_clip = 5
        # strength of dropout in the Language Model RNN
        self.drop_prob_lm = 0.5
        # After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)
        self.finetune_cnn_after = -1
        # number of captions to sample for each image during training.
        self.seq_per_img = 1
        # max number of iterations to run for (-1 = run forever)
        self.max_iters = -1
        # what update to use? rmsprop|sgd|sgdmom|adagrad|adam
        self.optim = 'adam'
        self.learning_rate = 0.0004
        # at what iteration to start decaying learning rate? (-1 = dont)
        self.learning_rate_decay_start = -1
        # every how many iterations thereafter to drop LR by half?
        self.learning_rate_decay_every = 50000
        # alpha for adagrad/rmsprop/momentum/adam
        self.optim_alpha = 0.8
        # beta used for adam
        self.optim_beta = 0.999
        # epsilon that goes into denominator for smoothing
        self.optim_epsilon = 1e-8
        # number of LSTM layers
        self.num_layers = 1
        # optimization to use for CNN
        self.cnn_optim = 'adam'
        # learning rate for the CNN
        self.cnn_learning_rate = 1e-5
        # L2 weight decay just for the CNN
        self.cnn_weight_decay = 0
        # alpha for momentum of CNN
        self.cnn_optim_alpha = 0.8
        # beta for momentum of CNN
        self.cnn_optim_beta = 0.999
        # how many images to use when periodically evaluating the validation loss? (-1 = all)
        self.val_images_use = 500
        # how often to save a model checkpoint?
        self.save_checkpoint_every = 500
        # folder to save checkpoints into (empty = this folder)
        self.checkpoint_path = 'snapshot'
        # How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)
        self.losses_log_every = 25
        # nn|cudnn
        self.backend = 'cudnn'
        # an id identifying this run/job. used in cross-val and appended when writing progress files
        self.id = ''
        self.logger.info(self.__dict__)
