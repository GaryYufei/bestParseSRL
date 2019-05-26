import json
from io_utils import *
import numpy as np

class Config:

	def add_config(self, value_dict):
		for k, v in value_dict.items():
			setattr(self, k, v)

	def print_all_parameters(self):
		attrs = vars(self)
		print("---------------------------------")
		for k, v in attrs.items():
			if type(v) is dict or type(v).__module__ == np.__name__:
				continue
			print("%s : %s" % (k, str(v)))
		print("---------------------------------")

	def load_all_data_file(self):
		self.tag_dict = read_item_file(self.tag_file, add_pad=False)
		self.char_dict = read_item_file(self.char_set_file, add_pad=False)
		self.pos_dict = read_item_file(self.pos_set_file, add_pad=False)
		self.span_label_dict = read_item_file(self.span_label_file, add_pad=False) if self.use_span_label else constraint_index_mapping
		self.relative_layer_dict = read_item_file(self.relative_layer_file, add_pad=False)
		self.ancestors_label_dict = read_item_file(self.ancestors_label_file, add_pad=False)
		self.dp_label_dict = read_item_file(self.dp_label_file, add_pad=False)
		self.word_embedding, self.word_dict = create_word_embedding(self.embedding_file, self.word_embeds_dim)

		self.nchars = len(self.char_dict)
		self.ntags = len(self.tag_dict)
		self.nwords = len(self.word_dict)
		self.npos = len(self.pos_dict)
		self.nspan = len(self.span_label_dict)
		self.n_rel_label = len(self.relative_layer_dict)
		self.n_ancestors_label = len(self.ancestors_label_dict)
		self.n_dp_label = len(self.dp_label_dict)

	def read_dataset(self, srl, parse, dp, pred_parse, pred_dp, data_type):
		data_set = DataReader(self, self.word_dict, self.tag_dict, self.char_dict, self.pos_dict, self.span_label_dict, self.relative_layer_dict, self.ancestors_label_dict, self.dp_label_dict, data_type)
		data_set.load_data(srl, parse, dp, pred_parse, pred_dp)
		return data_set

def load_json(path):
	with open(path) as input_json:
		_config = json.loads(input_json.read())
	return _config

def make_config(config_path):
	overall_config = load_json(config_path)
	config = Config()

	model_config = load_json(overall_config['model'])

	config.add_config(model_config)
	data_dict = {}
	data_config = load_json(overall_config['data'])
	for k, v in data_config.items():
		if type(v) is str:
			data_dict[k] = v
		elif type(v) is list:
			for data_info in v:
				for sub_k, sub_v in data_info.items():
					key_info = k + "_" + sub_k
					if key_info not in data_dict:
						data_dict[key_info] = []
					data_dict[key_info].append(sub_v)
		elif type(v) is dict:
			for sub_k, sub_v in v.items():
				data_dict[k + "_" + sub_k] = sub_v
		else:
			raise ValueError('Invalid ' + str(v))
	for k, v in overall_config.items():
		if k in ['model', 'data']:
			continue
		data_dict[k] = v
	config.add_config(data_dict)
	return config
