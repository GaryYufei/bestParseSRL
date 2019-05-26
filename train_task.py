from model_task import SRL_LSTM
from config import *
import random
import numpy as np
import argparse
from sys import argv

running_command = 'python ' + ' '.join(argv)
random_seed = 20170930
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='config to run')
parser.add_argument('-debug',  help='enable output system info', action="store_true")
args = parser.parse_args()

_config = make_config(os.path.join(args.model_dir, "run.json"))
_config.load_all_data_file()

is_dev = False
use_nbest = 1
model = SRL_LSTM(_config, use_nbest)
model.build_graph()

checkpoint_dir = os.path.join(args.model_dir, "checkpoint")
model.m_saver = ModelSaver(checkpoint_dir, args.debug, is_dev)

srl_dev = _config.read_dataset(_config.dev_srl, _config.dev_parse, _config.dev_dp, _config.dev_pred_parse, _config.dev_pred_dp, _config.dev_data_type)
srl_train = _config.read_dataset(_config.train_srl, _config.train_parse, _config.train_dp, _config.train_pred_parse, _config.train_pred_dp, _config.train_data_type)

print("Vocab Size {}".format(len(_config.word_dict)))
print("OOV: Train {:.2} Dev {:.2}".format(srl_train.get_OOV_ratio() * 100, srl_dev.get_OOV_ratio() * 100))
print("running script: %s" % running_command)
_config.print_all_parameters()

with tf.Session() as sess:
	sess.run(model.init)
	model.train(sess, srl_train, srl_dev)
