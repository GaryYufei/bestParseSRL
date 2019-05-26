import tensorflow as tf
import operator
import random
from os import path
from math import pow
import numpy as np
from tqdm import tqdm
import sys
from io_utils import *
from structure import *
from stack_LSTM import deep_biLSTM

def conv1d(in_, filter_size, height, padding, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        in_ = tf.nn.dropout(in_, keep_prob)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
        return out

def char_cnn(in_, size_list, height_list, padding, keep_prob=1.0):
	outs = []
	for filter_size, height in zip(size_list, height_list):
		out = conv1d(in_, filter_size, height, padding, keep_prob=keep_prob, scope="char_{}".format(height))
		outs.append(out)
	return tf.concat(outs, axis=-1)

def setup_elmo(sentences, text_len, elmo_path):
	import tensorflow_hub as hub
	elmo = hub.Module(elmo_path, trainable=True)
	return elmo(
	    inputs={
	        "tokens": sentences,
	        "sequence_len": text_len
	    },
	    signature="tokens",
	    as_dict=True)["elmo"]

def calculate_feature_loss(network_final_rep, labels, sequence_lengths, ntags, name):
	with tf.variable_scope('%s_loss' % name, initializer=tf.contrib.layers.xavier_initializer()):
		logits = tf.layers.dense(
				network_final_rep, ntags,
				kernel_initializer=tf.contrib.layers.xavier_initializer()
			)
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		mask = tf.sequence_mask(sequence_lengths, name="seq_mask", dtype=tf.int32)
		split_output = tf.dynamic_partition(losses, mask, 2)
		loss = tf.reduce_mean(split_output[1])
		prediction = tf.argmax(logits, axis=-1)

		return loss, prediction

def get_embedding_operation(labels, n_embedding, embedding_dim, name):
	_all_embeddings = tf.get_variable("_%s_embeddings" % name, dtype=tf.float32, shape=[n_embedding, embedding_dim])
	return tf.nn.embedding_lookup(_all_embeddings, labels, name="%s_embeddings" % name)

def GCN(input_vec, dep_heads):
	hidden_size = input_vec.get_shape().as_list()[2]
	parent_onehot = tf.one_hot(dep_heads, tf.shape(input_vec)[1])
	children_onehot = tf.transpose(parent_onehot, perm=[0, 2, 1])

	parent_vec = tf.matmul(parent_onehot, input_vec)
	children_vec = tf.matmul(children_onehot, input_vec)

	input_vec = tf.layers.dense(input_vec, hidden_size, name="self_dense")
	children_vec = tf.layers.dense(children_vec, hidden_size, name="children_dense")
	parent_vec = tf.layers.dense(parent_vec, hidden_size, name="parent_dense")
	final_rep = tf.concat([input_vec, children_vec, parent_vec], axis=-1)

	return tf.nn.relu(final_rep)

def srl_loss_predict(in_, labels, sequence_lengths, ntags, dep_heads=None):
	if dep_heads is not None:
		batch_size, length = tf.shape(in_)[0], tf.shape(in_)[1]
		batch_idx = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, length]), axis=-1)
		head_idx = tf.concat([batch_idx, tf.expand_dims(dep_heads, axis=-1)], axis=-1)
		selected_head_rep = tf.gather_nd(in_, head_idx)
		sytactic_attention = tf.concat([in_, selected_head_rep], axis=-1)
		# sytactic_attention = GCN(in_, dep_heads)
	else:
		sytactic_attention = in_

	logits = tf.layers.dense(
		sytactic_attention, ntags,
		kernel_initializer=tf.contrib.layers.xavier_initializer()
	)

	log_likelihood, transition_params = \
			tf.contrib.crf.crf_log_likelihood(logits, labels, sequence_lengths)
	loss = tf.reduce_mean(-log_likelihood)

	return loss, logits, transition_params

def dp_loss(network_final_rep, dep_heads, sequence_lengths, dp_hidden_size):
	with tf.variable_scope("network_rep", initializer=tf.contrib.layers.xavier_initializer()):
		head_rep = tf.layers.dense(
			network_final_rep, dp_hidden_size,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),
			name='head_dense'
		)
		tail_rep = tf.layers.dense(
			network_final_rep, dp_hidden_size,
			kernel_initializer=tf.contrib.layers.xavier_initializer(),
			name='tail_dense'
		)
		head_rep_label, head_rep_dep = tf.split(head_rep, 2, axis=-1)
		tail_rep_label, tail_rep_dep = tf.split(tail_rep, 2, axis=-1)

	
	length = tf.shape(head_rep_dep)[1]
	batch_size = tf.shape(head_rep_dep)[0]

	with tf.variable_scope("MST_loss", initializer=tf.contrib.layers.xavier_initializer()):

		head_dense = tf.tile(tf.expand_dims(head_rep_dep, 2), [1, 1, length, 1])
		tail_dense = tf.tile(tf.expand_dims(tail_rep_dep, 1), [1, length, 1, 1])

		input_for_score = tf.tanh(head_dense + tail_dense)
		arc_score = tf.layers.dense(
				input_for_score, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='MLP_score'
			)
		arc_score = tf.reshape(arc_score, shape=(-1, length, length))
		
		self_mask = tf.logical_not(tf.eye(length, dtype=tf.bool))
		head_mask = tf.sequence_mask(tf.tile(tf.expand_dims(sequence_lengths, 1), [1, length]))
		zero_mask = tf.logical_and(head_mask, self_mask)
		softmax_mask = tf.cast(tf.logical_not(zero_mask), dtype=tf.float32) * tf.constant(-1 * 1e30, dtype=tf.float32)
		mst_score = tf.nn.softmax(arc_score * tf.cast(zero_mask, dtype=tf.float32) + softmax_mask)

		head_one_hot = tf.one_hot(dep_heads, length, dtype=tf.float32)
		mask = tf.sequence_mask(sequence_lengths)
		root_mask = tf.sequence_mask(tf.ones(tf.shape(sequence_lengths)), maxlen=length)
		mask = tf.cast(tf.logical_and(tf.logical_not(root_mask), mask), dtype=tf.float32)
		target_mst_score = tf.log(tf.reduce_sum(mst_score * head_one_hot, axis=-1) + tf.cast(root_mask, dtype=tf.float32)) * mask
		mst_loss = -1.0 * tf.reduce_mean(tf.reduce_sum(target_mst_score, axis=-1))

	return mst_loss

class SRL_LSTM(object):

	def __init__(self, config, nbest):
		self.config = config
		self.use_parse = self.config.parse or self.config.use_dp_loss
		self.m_saver = None
		self.use_elmo = self.config.elmo
		self.constrain = self.config.soft_dp
		self.use_hard_dp = self.config.hard_dp
		self.nbest = nbest
		self.graph_decoding = not (nbest > 1 or self.config.hard_dp)
		self.reversed_tag_dict = {v: k for (k, v) in self.config.tag_dict.items()}
		self.tag_dict = self.config.tag_dict
		self.use_tree_encode = self.config.use_tree_encode
		self.use_dp_tree_feature = self.config.use_dp_tree_feature

		if self.use_hard_dp:
			self.trans_mask_array, self.unary_mask_array = create_trans_mask(self.reversed_tag_dict)

		if not self.use_elmo:
			self.input = tf.placeholder(tf.int32, shape=[None, None],
	                        name="input_sentence")
			assert self.config.word_embedding.shape[1] == config.word_embeds_dim
			self.pretrain_embedds = self.config.word_embedding
		else:
			self.input = tf.placeholder(tf.string, shape=[None, None],
	                        name="input_sentence")

		self.verb_pos = tf.placeholder(tf.int32, shape=[None, None],
                        name="verb_pos")
		self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")
		self.part_of_speech = tf.placeholder(tf.int32, shape=[None, None],
                        name="part_of_speech")
		self.chars = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="chars")
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")
			
		self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
		#if self.use_elmo:
		self.input_dropout = tf.placeholder(dtype=tf.float32, shape=[], name="input_dropout")

		if self.constrain or self.config.feature_loss:
			self.constrain_index = tf.placeholder(dtype=tf.int32, shape=[None, None], name="constrain_index")

		if self.use_parse:
			self.dep_labels = tf.placeholder(tf.int32, shape=[None, None], name="dep_labels")

		if self.use_tree_encode or self.config.tree_encode_loss:
			self.relative_layer = tf.placeholder(tf.int32, shape=[None, None], name="relative_layer")
			self.ancestors_label = tf.placeholder(tf.int32, shape=[None, None], name="ancestors_label")

		if self.use_dp_tree_feature or self.config.use_dp_tree_feature_loss:
			self.c_type = tf.placeholder(tf.int32, shape=[None, None], name="c_type")
			self.left_ch_num = tf.placeholder(tf.int32, shape=[None, None], name="left_ch_num")
			self.right_ch_num = tf.placeholder(tf.int32, shape=[None, None], name="right_ch_num")
			self.rel_parent_pos = tf.placeholder(tf.int32, shape=[None, None], name="rel_parent_pos")
			self.pp_type = tf.placeholder(tf.int32, shape=[None, None], name="pp_type")
			self.dp_labels = tf.placeholder(tf.int32, shape=[None, None], name="dp_label")

	def get_feed_dict(self, seq_len, verb_pos, batch_sen, batch_char, batch_pos, constrain_idx, relative_layer, ancestors_label, batch_dep_ct, batch_left_ch_num, batch_right_ch_num, batch_dep_ppt, batch_rel_parent_pos, batch_dp_labels, 
			batch_y=None, dep_label=None, dropout=0.0, input_dropout=0.0):
		feed = dict()

		feed[self.input] = batch_sen
		feed[self.verb_pos] = verb_pos
		feed[self.sequence_lengths] = seq_len
		feed[self.chars] = batch_char
		feed[self.part_of_speech] = batch_pos

		if self.constrain or self.config.feature_loss:
			feed[self.constrain_index] = constrain_idx

		if batch_y is not None:
			feed[self.labels] = batch_y
		if self.use_parse:
			feed[self.dep_labels] = dep_label

		feed[self.dropout] = dropout
		feed[self.input_dropout] = input_dropout

		if self.use_tree_encode or self.config.tree_encode_loss:
			feed[self.relative_layer] = relative_layer
			feed[self.ancestors_label] = ancestors_label

		if self.use_dp_tree_feature or self.config.use_dp_tree_feature_loss:
			feed[self.c_type] = batch_dep_ct
			feed[self.left_ch_num] = batch_left_ch_num
			feed[self.right_ch_num] = batch_right_ch_num
			feed[self.rel_parent_pos] = batch_rel_parent_pos
			feed[self.pp_type] = batch_dep_ppt
			feed[self.dp_labels] = batch_dp_labels

		return feed

	def train(self, sess, train_data, dev_data):
		for epoch in range(1, self.config.nepoch + 1):
			print("Epoch %d" % epoch)
			with tqdm(total=len(train_data), file=sys.stdout) as pbar:
				for (batch_len, batch_position, batch_sen, batch_tag, barch_char, batch_pos, batch_dep_heads, batch_trans_mask, batch_relative_layer, batch_ancestors_label, batch_dep_ct, batch_left_ch_num, batch_right_ch_num, batch_dep_ppt, batch_rel_parent_pos, batch_dp_labels) in train_data:
					fd = self.get_feed_dict(batch_len, batch_position, batch_sen, barch_char, batch_pos, 
						batch_trans_mask, batch_relative_layer, batch_ancestors_label, batch_dep_ct, batch_left_ch_num, batch_right_ch_num, batch_dep_ppt, batch_rel_parent_pos, batch_dp_labels, batch_y=batch_tag, dep_label=batch_dep_heads, \
						dropout=self.config.dropout, input_dropout=self.config.elmo_dropout)
					if epoch < self.config.change_opt_epoch:
						[_, train_loss] = sess.run([self.adaedelta_train_op, self.loss], feed_dict=fd)
					else:
						[_, train_loss] = sess.run([self.sgd_train_op, self.loss], feed_dict=fd)
					pbar.set_description('loss %.4f' % (train_loss))
					pbar.update(1)
			new_ep = self.task_eval(sess, dev_data, self.config.dev_eval)
			self.m_saver.save(new_ep.F1, sess, epoch)
			self.m_saver.output_best_performance()

	def task_eval(self, sess, data, golden_file, output_file=None):
		new_viterbi_sequences = self.prediction(sess, data)
		if self.nbest == 1:
			f1 = print_srl_eval(new_viterbi_sequences, golden_file, self.config.eval_script, output_file)
		else:
			viterbi_sequences = []
			for nbest_seq, _, g_seq in new_viterbi_sequences:
				max_f1, max_seq = 0, None
				for i, q_seq in enumerate(nbest_seq):
					f1 = select_oracle(g_seq, q_seq, self.tag_dict)
					if f1 > max_f1:
						max_f1 = f1
						max_seq = q_seq
				viterbi_sequences.append([self.reversed_tag_dict[c] for c in max_seq[1:]])
			f1 = print_srl_eval(viterbi_sequences, golden_file, self.config.eval_script, output_file)
		return f1

	def prediction(self, sess, data):
		viterbi_sequences = []
		with tqdm(total=len(data), file=sys.stdout) as pbar:
			for batch_len, batch_position, batch_sen, batch_tag, barch_char, batch_pos, batch_dep_heads, batch_trans_mask, batch_relative_layer, batch_ancestors_label, batch_dep_ct, batch_left_ch_num, batch_right_ch_num, batch_dep_ppt, batch_rel_parent_pos, batch_dp_labels in data:
				fd = self.get_feed_dict(batch_len, batch_position, batch_sen, 
					barch_char, batch_pos, batch_trans_mask, batch_relative_layer, batch_ancestors_label, 
					batch_dep_ct, batch_left_ch_num, batch_right_ch_num, batch_dep_ppt, batch_rel_parent_pos, batch_dp_labels, dep_label=batch_dep_heads)
				if not self.graph_decoding:
					srl_logits, srl_tp = sess.run([self.srl_logits, self.transition_params], feed_dict=fd)
					# iterate over the sentences
					for ground_tag, mask_index, srl_logit, sequence_length in zip(batch_tag, batch_trans_mask, srl_logits, batch_len):
						# SRL eval
						if self.use_hard_dp:
							trans_mask = [self.trans_mask_array[v] for v in mask_index[:sequence_length]]
							unary_mask = [self.unary_mask_array[v] for v in mask_index[:sequence_length]]
							srl_viterbi_, _ = viterbi_decode(srl_logit[:sequence_length], srl_tp, trans_mask, unary_mask)
							viterbi_sequences.append([self.reversed_tag_dict[c] for c in srl_viterbi_[1:]])
						elif self.nbest > 1:
							srl_viterbi_nbest, seq_score = viterbi_decode_nbest(srl_logit[:sequence_length], srl_tp, self.nbest)
							viterbi_sequences.append((srl_viterbi_nbest, seq_score, ground_tag[:sequence_length]))
						else:
							raise ValueError('decoding method not supported')
				else:
					[srl_prediction] = sess.run([self.decode_tags], feed_dict=fd)
					for srl_p, sequence_length in zip(srl_prediction, batch_len):
						viterbi_sequences.append([self.reversed_tag_dict[c] for c in srl_p[1:sequence_length]])
				pbar.update(1)
		new_viterbi_sequences = [None for _ in range(len(viterbi_sequences))]
		for seq, k in zip(viterbi_sequences, data.sorted_order):
			new_viterbi_sequences[k] = seq
		return new_viterbi_sequences

	def build_graph(self):
		if self.use_dp_tree_feature or self.config.use_dp_tree_feature_loss:
			input_list = [self.left_ch_num, self.right_ch_num, self.c_type, self.rel_parent_pos, self.dp_labels]
			nspan_list = [RELATIVE_POSITION_RANGE * 2, RELATIVE_POSITION_RANGE * 2, 3, RELATIVE_POSITION_RANGE * 2, self.config.n_dp_label]
			name_list = ['left_children', 'right_children', 'child_type', 'rel_parent_distance', 'dep_label']
		
		with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
			if not self.use_elmo:
				# word level information
				word_params = tf.Variable(self.pretrain_embedds, dtype=tf.float32, name="SRL_embeddings")
				word_representation = tf.nn.embedding_lookup(word_params, self.input, name="word_embeddings")
			else:
				word_representation = setup_elmo(self.input, self.sequence_lengths, self.config.elmo_path)
				word_representation = tf.nn.dropout(word_representation, 1 - self.input_dropout)

			#char embedding
			_char_embeddings = tf.get_variable("_char_embeddings", dtype=tf.float32, shape=[self.config.nchars, self.config.char_embeds_dim])
			char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.chars, name="char_embeddings")
			final_char_embeddings = char_cnn(char_embeddings, [25, 25, 25, 25], [2,3,4,5], "VALID")

			#part of speech
			# _pos_embeddings = tf.get_variable("_pos_embeddings", dtype=tf.float32, 
			# 	shape=[self.config.npos, self.config.pos_embeds_dim])
			# pos_embeddings = tf.nn.embedding_lookup(_pos_embeddings, self.part_of_speech, name="pos_embeddings")

			if not self.constrain:
				# position embedding
				_position_embeddings = tf.get_variable("_position_embeddings", dtype=tf.float32, shape=[2, self.config.position_embeds_dim])
				constrain_embeds = tf.nn.embedding_lookup(_position_embeddings, self.verb_pos, name="position_embeddings")
			else:
				# constrain embedding
				_constrain_embeddings = tf.get_variable("_constrain_embeddings", dtype=tf.float32, shape=[self.config.nspan, self.config.position_embeds_dim])
				constrain_embeds = tf.nn.embedding_lookup(_constrain_embeddings, self.verb_pos, name="_constrain_embeds")

			# final combination
			self.input_representation = tf.concat([word_representation, final_char_embeddings, constrain_embeds], axis=-1)
			
			if self.use_tree_encode:

				if self.config.enable_relative_layer:
					_relative_layer_embeddings = tf.get_variable("_relative_layer_embeddings", dtype=tf.float32, shape=[self.config.n_rel_label, self.config.pos_embeds_dim])
					relative_layer_embeddings = tf.nn.embedding_lookup(_relative_layer_embeddings, self.relative_layer, name="pos_embeddings")
					self.input_representation = tf.concat([self.input_representation, relative_layer_embeddings], axis=-1)

				if self.config.enable_ancestor_label:
					_ancestors_label_embeddings = tf.get_variable("_ancestors_label_embeddings", dtype=tf.float32, shape=[self.config.n_ancestors_label, self.config.pos_embeds_dim])
					ancestors_label_embeddings = tf.nn.embedding_lookup(_ancestors_label_embeddings, self.ancestors_label, name="pos_embeddings")
					self.input_representation = tf.concat([self.input_representation, ancestors_label_embeddings], axis=-1)
		
			if self.use_dp_tree_feature:
				all_embeddings_list = [self.input_representation]
				for input_index, num_embedding, embedding_name in zip(input_list, nspan_list, name_list):
					all_embeddings_list.append(get_embedding_operation(input_index, num_embedding, self.config.dp_embeds_dim, embedding_name))

				self.input_representation = tf.concat(all_embeddings_list, axis=-1)

		with tf.variable_scope("SRL_LSTM", initializer=tf.contrib.layers.xavier_initializer()):
			input_size = [self.input_representation.get_shape().as_list()[-1]] + [self.config.hidden_size] * (self.config.layer_size - 1)
			output_rep_list, _ = deep_biLSTM(self.config.hidden_size, self.input_representation, self.sequence_lengths, 
				self.dropout, input_size)
			srl_model_output = output_rep_list[-1]

			dependency = self.dep_labels if self.config.parse else None
			srl_loss, self.srl_logits, self.transition_params = srl_loss_predict(srl_model_output, self.labels, self.sequence_lengths, 
				self.config.ntags, dep_heads=dependency)
			self.loss = srl_loss

			if self.config.pos_multi_task:
				task_loss, pred = calculate_feature_loss(srl_model_output, self.part_of_speech, 
					self.sequence_lengths, self.config.npos, "pos_feature")
				self.loss += task_loss

			if self.config.feature_loss:
				task_loss, pred = calculate_feature_loss(srl_model_output, self.constrain_index, 
					self.sequence_lengths, self.config.nspan, "span_feature")
				self.loss += task_loss

			if self.config.tree_encode_loss:
				task_loss, pred = calculate_feature_loss(srl_model_output, self.relative_layer, 
					self.sequence_lengths, self.config.n_rel_label, "relative_layer")
				self.loss += task_loss

				task_loss, pred = calculate_feature_loss(srl_model_output, self.ancestors_label, 
					self.sequence_lengths, self.config.n_ancestors_label, "ancestors_label")
				self.loss += task_loss

			if self.config.use_dp_tree_feature_loss:
				for input_idx, nspan, name in zip(input_list, nspan_list, name_list):
					task_loss, pred = calculate_feature_loss(srl_model_output, input_idx, self.sequence_lengths, nspan, name)
					self.loss += task_loss

			if self.config.use_dp_loss:
				mst_loss = dp_loss(srl_model_output, self.dep_labels, self.sequence_lengths, self.config.hidden_size * 2)
				self.loss += mst_loss
					
			if self.graph_decoding:
				self.decode_tags, _ = tf.contrib.crf.crf_decode(self.srl_logits, self.transition_params, self.sequence_lengths)

			sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.SGD_lr)
			grads = sgd_optimizer.compute_gradients(self.loss)
			for i, (g, v) in enumerate(grads):
				if g is not None:
					grads[i] = (tf.clip_by_norm(g, self.config.clip_norm), v)
			self.sgd_train_op = sgd_optimizer.apply_gradients(grads)
			
			# Adam delta for SRL
			adaedelta_optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.adam_delta_lr, 
				epsilon=self.config.epsilon)
			grads = adaedelta_optimizer.compute_gradients(self.loss)
			for i, (g, v) in enumerate(grads):
				if g is not None:
					grads[i] = (tf.clip_by_norm(g, self.config.clip_norm), v)
			self.adaedelta_train_op = adaedelta_optimizer.apply_gradients(grads)

		self.init = tf.global_variables_initializer()
