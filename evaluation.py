# This file is for ensemble and k-best prediction
from model_task import SRL_LSTM
from config import make_config
from io_utils import *
import argparse
import numpy as np
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Current model tag to save')
args = parser.parse_args()

_config = make_config(os.path.join(args.model_dir, "run.json"))
_config.load_all_data_file()

dp_list = [_config.dev_dp] + _config.test_dp
pred_dp_list = [_config.dev_pred_dp] + _config.test_pred_dp
parse_list = [_config.dev_parse] + _config.test_parse
pred_parse_list = [_config.dev_pred_parse] + _config.test_pred_parse
srl_list = [_config.dev_srl] + _config.test_srl
gold_list = [_config.dev_eval] + _config.test_eval

is_dev = True
is_debug = False
use_nbest = 1
model = SRL_LSTM(_config, use_nbest)
model.build_graph()

performance_dict = {'pred': {}, 'gold': {}}
with tf.Session() as sess:
	checkpoint_dir = os.path.join(args.model_dir, "checkpoint")
	model.m_saver = ModelSaver(checkpoint_dir, is_debug, is_dev)
	model.m_saver.load_model(sess)
	
	for dp, pred_dp, parse, pred_parse, test_file, test_gold_props_file in zip(dp_list, pred_dp_list, parse_list, pred_parse_list, srl_list, gold_list):
		srl_test = _config.read_dataset(test_file, parse, dp, pred_parse, pred_dp, 'pred')
		print("{} OOV: {:.2}".format(test_file, srl_test.get_OOV_ratio() * 100))

		performance = model.task_eval(sess, srl_test, test_gold_props_file)
		performance_dict['pred'][test_file] = performance.toDict()

		srl_test.what_to_use = 'gold'
		performance = model.task_eval(sess, srl_test, test_gold_props_file)
		performance_dict['gold'][test_file] = performance.toDict()

	prediction_path = os.path.join(args.model_dir, "result_pred.json")
	with open(prediction_path, 'w') as save:
		save.write(json.dumps(performance_dict['pred']))

	prediction_path = os.path.join(args.model_dir, "result_gold.json")
	with open(prediction_path, 'w') as save:
		save.write(json.dumps(performance_dict['gold']))
