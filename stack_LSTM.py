import tensorflow as tf

def get_LSTM_cell(hidden_size, dropout, input_size, with_connection):
	cell = tf.contrib.rnn.LSTMCell(hidden_size)
	if with_connection: cell = tf.contrib.rnn.ResidualWrapper(cell)
	cell = tf.contrib.rnn.DropoutWrapper(cell, 
		state_keep_prob=1.0 - dropout, 
		input_keep_prob=1.0 - dropout,
	 	variational_recurrent=True, dtype=tf.float32, input_size=input_size)
	return cell

def word_biLSTM(hidden_size, input_vec, sequence_lengths, dropout, input_size_list):
	forward_cell_list = [get_LSTM_cell(hidden_size, dropout, input_size, False) for input_size in input_size_list]
	backward_cell_list = [get_LSTM_cell(hidden_size, dropout, input_size, False) for input_size in input_size_list]

	(outputs, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
		forward_cell_list, backward_cell_list, input_vec,
		sequence_length=sequence_lengths, dtype=tf.float32)

	return outputs

def deep_biLSTM(hidden_size, input_vec, sequence_lengths, dropout, input_size_list):
	layer_input = input_vec
	output_list = []
	output_state_list = []

	for i, input_size in enumerate(input_size_list):
		LSTM_cell = get_LSTM_cell(hidden_size, dropout, input_size, i > 0)
		(layer_out, layer_state) = tf.nn.dynamic_rnn(LSTM_cell, layer_input, 
			sequence_length=sequence_lengths, scope="rnn_layer_%d" % i, dtype=tf.float32)
		layer_input = tf.reverse_sequence(layer_out, sequence_lengths, 
			seq_axis=1, name='seq_reverse_%d' % i)
		output_list.append(layer_input)

	return output_list, output_state_list