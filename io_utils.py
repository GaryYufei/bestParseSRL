import random
import numpy as np
import time
import sys
from string import punctuation
import tensorflow as tf
import subprocess
import math
import os
from annotation import *
from tqdm import tqdm
import tempfile

def create_trans_mask(reversed_tag):
	ntags = len(reversed_tag)
	trans_mask = np.ones((6, ntags, ntags))
	unary_mask = np.ones((6, ntags))
	Inf = 1e10
	for i in range(ntags):
		for j in range(ntags):
			if reversed_tag[j].startswith('B') or reversed_tag[j].startswith('I') or reversed_tag[j] in ['B-V', 'I-V']:
				trans_mask[0, i, j] = -1 * Inf
				unary_mask[0, j] = -1 * Inf
	for i in range(ntags):
		for j in range(ntags):
			if reversed_tag[j].startswith('I') or  reversed_tag[j] in ['B-V', 'I-V']:
				trans_mask[1, i, j] = -1 * Inf
				unary_mask[1, j] = -1 * Inf
	for i in range(ntags):
		for j in range(ntags):
			if reversed_tag[j].startswith('B') or reversed_tag[j] in ['B-V', 'I-V']:
				trans_mask[2, i, j] = -1 * Inf
				unary_mask[2, j] = -1 * Inf
	for i in range(ntags):
		for j in range(ntags):
			if reversed_tag[j] == 'B-V':
				continue
			trans_mask[4, i, j] = -1 * Inf
			unary_mask[4, j] = -1 * Inf
	for i in range(ntags):
		for j in range(ntags):
			if reversed_tag[j] == 'I-V':
				continue
			trans_mask[5, i, j] = -1 * Inf
			unary_mask[5, j] = -1 * Inf
	return trans_mask, unary_mask

def get_chunks(seq, tags, default_tag):

    def get_chunk_type(tok, idx_to_tag):
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type
        
    default = tags[default_tag]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def select_oracle(golden, prediction, tag_dict, default_tag='O'):
	gold_arguments = get_chunks(golden, tag_dict, default_tag)
	pred_arguments = get_chunks(prediction, tag_dict, default_tag)

	if len(pred_arguments) == 0 or len(gold_arguments) == 0:
		return 0.0

	correct = 0
	for p in pred_arguments:
		for g in gold_arguments:
			if p[0] == g[0] and p[1] == g[1] and p[2] == g[2]:
				correct += 1
				break
	precision = correct / len(pred_arguments)
	recall = correct / len(gold_arguments)

	return 2 * precision * recall / (recall + precision) if correct > 0 else 0.0

def read_item_file(file_name, add_pad=True):
	tag_dict = {}
	if add_pad:
		tag_dict['<PAD>'] = 0
	with open(file_name) as tag_f:
		for line in tag_f:
			line = line.strip()
			if line not in tag_dict:
				tag_dict[line] = len(tag_dict)
	return tag_dict

def read_word_embedding(file_name):
	word_embedding_word = []
	word_embedding_dict = {}
	with open(file_name) as word_embedding:
		for line in word_embedding:
			line = line.strip()
			current_word = line.split()[0]
			word_embedding_word.append(current_word)
			word_embedding_dict[current_word] = line.split()[1:]
	return word_embedding_word, word_embedding_dict

def create_word_embedding(embedding_file, embeds_dim):
	word_list, word_dict = read_word_embedding(embedding_file)
	word_index_dict = {UNKNOWN_WORD: 1, PAD: 0, ROOT: 2}
	word_embedding = np.random.normal(scale=0.01, size=(len(word_dict) + len(word_index_dict), embeds_dim))
	for w in word_list:
		word_index_dict[w] = len(word_index_dict)
		assert len(word_dict[w]) == embeds_dim
		word_embedding[word_index_dict[w], :] = np.asarray(word_dict[w], dtype=np.float64)
	return word_embedding, word_index_dict

def bio_to_se(labels):
	slen = len(labels)
	new_labels = []
	for i, label in enumerate(labels):
		if label == 'O':
			new_labels.append('*')
			continue
		new_label = '*'
		if label[0] == 'B' or i == 0 or label[1:] != labels[i-1][1:]:
			new_label = '(' + label[2:] + new_label
		if i == slen - 1 or labels[i+1][0] == 'B' or label[1:] != labels[i+1][1:]:
			new_label = new_label + ')'
		new_labels.append(new_label)
	return new_labels

def print_sentence_to_conll(fout, tokens, labels):
	for label_column in labels:
		assert len(label_column) == len(tokens)
	for i in range(len(tokens)):
		fout.write(tokens[i].ljust(15))
		for label_column in labels:
			fout.write(label_column[i].rjust(15))
		fout.write("\n")
	fout.write("\n")
  
def print_to_conll(pred_labels, gold_props_file, output_filename):
	
	seq_ptr = 0
	num_props_for_sentence = 0
	tokens_buf = []
	
	with open(output_filename, 'w') as fout:
		with open(gold_props_file) as gold_file:
			for line in gold_file:
				line = line.strip()
				if line == "" and len(tokens_buf) > 0:
					pred_output = pred_labels[seq_ptr:seq_ptr+num_props_for_sentence]
					print_sentence_to_conll(fout, tokens_buf, pred_output)
					seq_ptr += num_props_for_sentence
					tokens_buf = []
					num_props_for_sentence = 0
				else:
					info = line.split('\t')
					num_props_for_sentence = len(info) - 1
					tokens_buf.append(info[0])
			
			# Output last sentence. 
			if len(tokens_buf) > 0:
				pred_output = pred_labels[seq_ptr:seq_ptr+num_props_for_sentence]
				print_sentence_to_conll(fout, tokens_buf, pred_output)

def print_srl_eval(srl_predictions, gold_props, eval_script, output_file=None):
	temp_dir = None
	if output_file is None:
		temp_dir = tempfile.TemporaryDirectory(prefix="srl_eval-")
		output_file = os.path.join(temp_dir.name, 'out.txt')
	viterbi_sequences = [bio_to_se(srl_prediction) for srl_prediction in srl_predictions]
	print_to_conll(viterbi_sequences, gold_props, output_file)
	child = subprocess.Popen('perl {} {} {}'.format(eval_script, gold_props, output_file), 
		shell = True, stdout=subprocess.PIPE)
	eval_output = child.communicate()[0].decode("utf-8")
	print(eval_output)
	if temp_dir is not None:
		temp_dir.cleanup()
	f1 = float(eval_output.strip().split("\n")[6].strip().split()[6])
	recall = float(eval_output.strip().split("\n")[6].strip().split()[5])
	precision = float(eval_output.strip().split("\n")[6].strip().split()[4])
	comp = float(eval_output.strip().split("\n")[2].strip().split()[5])
	ep = EvaluationPerformance(f1, recall, precision, comp)
	return ep

class EvaluationPerformance(object):

	def __init__(self, f1, recall, precision, comp):
		self.F1 = f1
		self.Recall = recall
		self.Precision = precision
		self.Comp = comp
	
	def toDict(self):
		return {'f1': self.F1, 'recall': self.Recall, 'precision': self.Precision, 'comp': self.Comp}

class DataReader(object):
	def __init__(self, config, input_word_dict, output_tag_dict, char_dict, pos_dict, span_label_dict, relative_level_dict, ancestors_label_dict, dep_label_dict, what_to_use, is_train=True):
		self.config = config
		self.input_word_dict = input_word_dict
		self.output_tag_dict = output_tag_dict
		self.oov_count = {'total': 0, 'oov': 0}
		self.batch_size = config.batch_size
		self.char_dict = char_dict
		self.pos_dict = pos_dict
		self.use_elmo = config.elmo
		self.pad_sen_func = tf.keras.preprocessing.sequence.pad_sequences
		self.large_batch_split = config.large_batch_split
		self.large_batch_size = 30
		self.span_label_dict = span_label_dict
		self.what_to_use = what_to_use
		self.is_train = is_train
		self.relative_level_dict = relative_level_dict
		self.ancestors_label_dict = ancestors_label_dict
		self.dep_label_dict = dep_label_dict
		self.batch_list = []

	def load_data(self, srl, parse, dp, pred_parse, pred_dp):
		# read golden
		verb_idx_list, idx_sen_list, tag_list, char_list, g_part_of_speech, dep_heads, g_trans_mask, g_relative_layer, ancestors_label, dep_child_type, dep_left_ch_num, dep_right_ch_num, dep_predicate_parent, dep_rel_gov_pos, batch_dp_labels = self.read_data(srl, parse, dp)
		self.golden_data = list(zip(verb_idx_list, idx_sen_list, tag_list, char_list, g_part_of_speech, dep_heads, g_trans_mask, g_relative_layer, ancestors_label, dep_child_type, dep_left_ch_num, dep_right_ch_num, dep_predicate_parent, dep_rel_gov_pos, batch_dp_labels))
		# read pred
		verb_idx_list, idx_sen_list, tag_list, char_list, part_of_speech, dep_heads, trans_mask, relative_layer, ancestors_label, dep_child_type, dep_left_ch_num, dep_right_ch_num, dep_predicate_parent, dep_rel_gov_pos, batch_dp_labels = self.read_data(srl, pred_parse, pred_dp)
		total, same = 0, 0
		for g, p in zip(g_trans_mask, trans_mask):
			for g_w, p_w in zip(g[1:], p[1:]):
				total += 1
				same += 1 if g_w == p_w else 0
		print("srl-cons %d / %d = %.2f" % (same, total, same * 100 / total))

		total, same = 0, 0
		for g, p in zip(g_relative_layer, relative_layer):
			for g_w, p_w in zip(g[1:], p[1:]):
				total += 1
				same += 1 if g_w == p_w else 0
		print("full-cons %d / %d = %.2f" % (same, total, same * 100 / total))


		if self.config.pos_multi_task:
			part_of_speech = g_part_of_speech
		if self.config.feature_loss:
			trans_mask = g_trans_mask
		self.pred_data = list(zip(verb_idx_list, idx_sen_list, tag_list, char_list, part_of_speech, dep_heads, trans_mask, relative_layer, ancestors_label, dep_child_type, dep_left_ch_num, dep_right_ch_num, dep_predicate_parent, dep_rel_gov_pos, batch_dp_labels))


		
	def to_word_ids(self, w):
		word = w.lower()
		self.oov_count['total'] += 1
		if word in self.input_word_dict:
			return self.input_word_dict[word]
		self.oov_count['oov'] += 1
		return self.input_word_dict[UNKNOWN_WORD]

	def get_OOV_ratio(self):
		return self.oov_count['oov'] / self.oov_count['total']

	def get_sen_key(self, sen):
		return ' '.join(sen[1:])

	def read_data(self, srl_filename, parse_filename, dp_filename):
		verb_idx_list, sen_list, tag_list = read_srl_annotation(srl_filename)
		dp_mapping = read_dp_annotation(dp_filename)
		dep_heads = []
		dep_labels = []
		dep_child_type = []
		dep_left_ch_num = []
		dep_right_ch_num = []
		dep_predicate_parent = []
		dep_rel_gov_pos = []
		with tqdm(total=len(sen_list)) as pbar:
			for tag, sen in zip(tag_list, sen_list):
				key = self.get_sen_key(sen)
				predicate_index = tag.index('B-V')
				head, g_label = dp_mapping[key]
				dp_head = [0] + head
				dp_label = ['ROOT'] + g_label				
				left_ch_n, right_ch_n, rel_f_pos, c_type, pp_type = [], [], [], [], []
				for i in range(len(sen)):
					num_left_ch, num_right_ch, relative_f_pos, predicate_as_parent, child_type = extract_dependency_feature(predicate_index, i, dp_head)
					left_ch_n.append(num_left_ch)
					right_ch_n.append(num_right_ch)
					rel_f_pos.append(relative_f_pos)
					c_type.append(child_type)
					pp_type.append(predicate_as_parent)
				dep_left_ch_num.append(left_ch_n)
				dep_right_ch_num.append(right_ch_n)
				dep_child_type.append(c_type)
				dep_predicate_parent.append(pp_type)
				dep_rel_gov_pos.append(rel_f_pos)
				dep_heads.append(dp_head)
				dep_labels.append(dp_label)
				pbar.update(1)
		print("dep done")

		parse_mapping = read_parse_annotation(parse_filename)
		trans_mask = []
		part_of_speech = []
		relative_layer = []
		ancestors_label = []
		with tqdm(total=len(sen_list)) as pbar:
			for tag, sen in zip(tag_list, sen_list):
				key = self.get_sen_key(sen)
				predicate_index = tag.index('B-V') - 1
				(parse_tree, (label, relative_l), transition_operations) = parse_mapping[key]
				constrains = ['R-PP' if self.config.use_span_label else 'R'] + get_constrain_seq(parse_tree, predicate_index)
				constrains = [t if t in ['B-V', 'I-V'] else c for (t, c) in zip(tag, constrains)]
				constrains = constrains if self.config.use_span_label else [t if t in ['B-V', 'I-V'] else t[0] for t in constrains]
				part_of_speech.append(['NN'] + [pos for (_, pos) in parse_tree.pos()])
				trans_mask.append(constrains)
				relative_layer.append(['NONE'] + relative_l)
				ancestors_label.append(['NONE'] + label)
				pbar.update(1)

		assert len(verb_idx_list) == len(sen_list) and len(tag_list) == len(sen_list)
		idx_sen_list = [[self.to_word_ids(w) for w in sen] for sen in sen_list]
		if self.use_elmo: idx_sen_list = sen_list
		tag_list = [[self.output_tag_dict[t] for t in tag] for tag in tag_list]
		char_list = [[[self.char_dict[c] for c in w] for w in sen] for sen in sen_list]
		part_of_speech = [[self.pos_dict[c] for c in sen] for sen in part_of_speech]
		dep_labels = [[self.dep_label_dict[c] for c in sen] for sen in dep_labels]
		trans_mask = [[self.span_label_dict[c] for c in sen] for sen in trans_mask]

		if self.config.soft_dp:
			verb_idx_list = trans_mask
		print("parse done")

		count = 0
		for sen in relative_layer:
			for c in sen:
				if c not in self.relative_level_dict:
					count += 1
		print("RELATIVE LAYER MISSING: %d" % count)

		count = 0
		for sen in ancestors_label:
			for c in sen:
				if c not in self.ancestors_label_dict:
					count += 1
		print("ANCESTORS LAYER MISSING: %d" % count)
		relative_layer = [[self.relative_level_dict[c] if c in self.relative_level_dict else self.relative_level_dict['NONE'] for c in sen] for sen in relative_layer]
		ancestors_label = [[self.ancestors_label_dict[c] if c in self.ancestors_label_dict else self.ancestors_label_dict['NONE'] for c in sen] for sen in ancestors_label]
		return verb_idx_list, idx_sen_list, tag_list, char_list, part_of_speech, dep_heads, trans_mask, relative_layer, ancestors_label, dep_child_type, dep_left_ch_num, dep_right_ch_num, dep_predicate_parent, dep_rel_gov_pos, dep_labels
		
	def has_next(self):
		return len(self.batch_list) > 0

	def __iter__(self):
		return self

	def choose_data(self):
		if self.what_to_use == 'gold':
			return self.golden_data
		elif self.what_to_use == 'pred':
			return self.pred_data
		elif self.what_to_use == 'mix':
			return [g if random.random() > 0.5 else p for (g, p) in zip(self.golden_data, self.pred_data)]
		else:
			raise ValueError('Unknown Data Type: {}'.format(self.what_to_use))

	def compute_batch(self):
		combined_data = self.choose_data()
		current_data = [(i, data) for i, data in enumerate(combined_data)]
		current_data = sorted(current_data, key=lambda x: (len(x[1][1]), random.random()))
		large_batch = [x for x in current_data if len(x[1][1]) >= self.large_batch_split]
		small_batch = [x for x in current_data if len(x[1][1]) < self.large_batch_split]
		indexed_batch_list = []
		for i in range(0, len(small_batch), self.batch_size):
			indexed_batch_list.append(small_batch[i : i + self.batch_size])
		for i in range(0, len(large_batch), self.large_batch_size):
			indexed_batch_list.append(large_batch[i : i + self.large_batch_size])
		self.sorted_order = []
		self.batch_list = []
		for batch in indexed_batch_list:
			new_batch = []
			for x in batch:
				new_batch.append(x[1])
				self.sorted_order.append(x[0])
			self.batch_list.append(new_batch)

	def __next__(self):
		if not self.has_next():
			raise StopIteration()

		if len(self.batch_list) == 0:
			self.compute_batch()

		data_step = self.batch_list.pop(0)
		[batch_verb_id, batch_sen_list, batch_tag_list, batch_char_list, batch_pos_list, batch_dep_heads, batch_trans_mask, batch_relative_layer, batch_ancestors_label, batch_child_type, batch_left_ch_num, batch_right_ch_num, batch_ppt, batch_rel_gov_pos, batch_dp_labels] = list(zip(*data_step))
		batch_len = np.array([len(x) for x in batch_sen_list])
		batch_size, max_len = batch_len.shape[0], np.max(batch_len)
		if not self.use_elmo:
			batch_sen = self.pad_sen_func(batch_sen_list, padding='post', truncating='post')
		else:
			max_len = np.max(batch_len)
			batch_sen = [x + [""] * (max_len - len(x)) for x in batch_sen_list]
		batch_tag = self.pad_sen_func(batch_tag_list, padding='post', truncating='post')
		batch_pos = self.pad_sen_func(batch_pos_list, padding='post', truncating='post')
		batch_trans_mask = self.pad_sen_func(batch_trans_mask, padding='post', truncating='post')
		batch_char_list = [bc + [[0]] * (len(batch_sen[0]) - len(bc)) for bc in batch_char_list]
		batch_char_lens = [[min(len(c), self.config.word_max_length) for c in bc] for bc in batch_char_list]
		batch_char = [self.pad_sen_func(bc, padding='post', truncating='post', maxlen=self.config.word_max_length) for bc in batch_char_list]
		batch_position = self.pad_sen_func(batch_verb_id, padding='post', truncating='post')
		batch_dep_heads = self.pad_sen_func(batch_dep_heads, padding='post', truncating='post')
		batch_relative_layer = self.pad_sen_func(batch_relative_layer, padding='post', truncating='post')
		batch_ancestors_label = self.pad_sen_func(batch_ancestors_label, padding='post', truncating='post')
		
		batch_child_type = self.pad_sen_func(batch_child_type, padding='post', truncating='post')
		batch_left_ch_num = self.pad_sen_func(batch_left_ch_num, padding='post', truncating='post')
		batch_right_ch_num = self.pad_sen_func(batch_right_ch_num, padding='post', truncating='post')
		batch_ppt = self.pad_sen_func(batch_ppt, padding='post', truncating='post')
		batch_rel_gov_pos = self.pad_sen_func(batch_rel_gov_pos, padding='post', truncating='post')
		batch_dp_labels = self.pad_sen_func(batch_dp_labels, padding='post', truncating='post')
		
		return batch_len, batch_position, batch_sen, batch_tag, batch_char, batch_pos, batch_dep_heads, batch_trans_mask, batch_relative_layer, batch_ancestors_label, batch_child_type, batch_left_ch_num, batch_right_ch_num, batch_ppt, batch_rel_gov_pos, batch_dp_labels

	def __len__(self):
		if len(self.batch_list) == 0: self.compute_batch()
		return len(self.batch_list)

class ModelSaver:

	def __init__(self, model_dir, debug, is_dev):
		self.serialization_dir = model_dir
		self.debug = debug or is_dev

		if not self.debug:
			if not os.path.exists(self.serialization_dir):
				os.makedirs(self.serialization_dir)
			else:
				raise ValueError('model dict should be empty')

		if not self.debug:
			self.running_out = os.path.join(self.serialization_dir, 'running_output.txt')
			sys.stdout = open(self.running_out, 'w')

		self.saver = tf.train.Saver(max_to_keep=1)
		self.best_metrics = 0
		self.iteration = 0
        
	def save(self, dev_metrics, sess, epoch):
		if dev_metrics >= self.best_metrics:
			print("best model ever")
			self.best_metrics = dev_metrics
			self.iteration = epoch
			if self.debug: return True
			model_path = os.path.join(self.serialization_dir, 'model')
			self.saver.save(sess, model_path, global_step=epoch)
			return True
		else:
			return False  

	def load_model(self, sess):
		file_to_load = tf.train.latest_checkpoint(self.serialization_dir)
		self.saver.restore(sess, file_to_load)

	def output_best_performance(self):
		print("Best model: dev %.2f in iteration %d" % (self.best_metrics, self.iteration))
