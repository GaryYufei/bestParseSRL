from nltk.tree import ParentedTree
from tree import SeqTree, RelativeLevelTreeEncoder
from lc import *
from tb import *
from math import ceil

UNKNOWN_WORD = "<UNK>"
ROOT = '<root>'
PAD = '<PAD>'
RELATIVE_POSITION_RANGE = 100
list_for_punctuation = ['.', '``', "''", ",", '-RRB-', '-LRB-', '-RCB-', '-LCB-', ':']
constraint_index_mapping = {
	'O': 0,
	'B': 1,
	'I': 2,
	'R': 3,
	'B-V': 4,
	'I-V': 5
}
transition_index_mapping = {
	'><': 0,
	'><]': 1,
	'>[': 2,
	'>(': 3,
	'[': 4
}

def process_word(token):
	if token in ['-RRB-', '-LRB-', '-RCB-', '-LCB-', '&lt;', '&gt;']:
		w = token
		w = w.replace('-RRB-', ')')
		w = w.replace('-LRB-', '(')
		w = w.replace('-RCB-', ')')
		w = w.replace('-LCB-', '(')
		w = w.replace('&lt;', '<')
		w = w.replace('&gt;', '>')
	else:
		w = token
		w = w.replace('{', '(')
		w = w.replace('}', ')')
		w = w.replace('\\', '')
		w = w.replace('\/', '/')
		w = w.replace('\*', '*')
	return w

def read_srl_annotation(file_name):
	verb_idx_list, sen_list, tag_list = [], [], []
	with open(file_name) as file_reader:
		for line in file_reader:
			items = line.strip().split('\t')
			verb_idx_list.append([0] + [int(d) for d in items[0].split()])
			sen = [process_word(w) for w in items[1].split()]
			sen_list.append([ROOT] + sen)
			tag_list.append(['O'] + items[2].split())
	return verb_idx_list, sen_list, tag_list

def read_dp_annotation(file_name):
	mapping = {}
	with open(file_name) as dp:
		words, heads, g_label = [], [], []
		for line in dp:
			line = line.strip()
			if len(line) == 0:
				mapping[' '.join(words)] = (heads, g_label)
				words, heads, g_label = [], [], []
			else:
				items = line.split()
				w = process_word(items[1])
				words.append(w)
				heads.append(int(items[6]))
				g_label.append(items[7])

	return mapping

def clip_value(value_to_clip, _max=RELATIVE_POSITION_RANGE - 1, _min=-1 * RELATIVE_POSITION_RANGE):
	if value_to_clip > _max:
		value_to_clip =  _max
	elif value_to_clip < _min:
		value_to_clip =  _min
	value_to_clip -= _min
	return value_to_clip

def extract_dependency_feature(predicate_index, word_index, parent_list):
	left_children_num = 0
	right_children_num = 0
	for i in range(0, len(parent_list)):
		if parent_list[i] == word_index:
			if i < word_index:
				left_children_num += 1
			elif i > word_index:
				right_children_num += 1

	left_children_num = clip_value(left_children_num)
	right_children_num = clip_value(right_children_num)
	relative_governor_position = clip_value(parent_list[word_index] - word_index)
	has_predicate_as_parent = 1 if parent_list[word_index] == predicate_index else 0

	# left most child?
	left_most_children = True
	for i in range(0, word_index):
		if parent_list[i] == parent_list[word_index]:
			left_most_children = False
	
	# right most child?
	right_most_children = True
	for i in reversed(list(range(word_index + 1, len(parent_list)))):
		if parent_list[i] == parent_list[word_index]:
			right_most_children = False

	child_type = 0
	if left_most_children:
		child_type = 1
	elif right_most_children:
		child_type = 2

	assert left_children_num >= 0 and left_children_num < RELATIVE_POSITION_RANGE * 2, "%d is not good left children num" % left_children_num
	assert right_children_num >= 0 and right_children_num < RELATIVE_POSITION_RANGE * 2, "%d is not good right children num" % left_children_num
	assert relative_governor_position >= 0 and relative_governor_position < RELATIVE_POSITION_RANGE * 2, "%d is not good relative gov position" % relative_governor_position 

	return left_children_num, right_children_num, relative_governor_position, has_predicate_as_parent, child_type

def old_extract_dependency_feature(predicate_index, word_index, parent_list):
	# left most child?
	left_most_children = True
	has_left_children = False
	for i in range(0, word_index):
		if parent_list[i] == parent_list[word_index]:
			left_most_children = False
			
		if parent_list[i] == word_index:
			has_left_children = True
	
	# right most child?
	right_most_children = True
	has_right_children = False
	for i in reversed(list(range(word_index + 1, len(parent_list)))):
		if parent_list[i] == parent_list[word_index]:
			right_most_children = False
		
		if parent_list[i] == word_index:
			has_right_children = True

	has_predicate_as_parent = 1 if parent_list[word_index] == predicate_index else 0

	child_type = 0
	if left_most_children:
		child_type = 1
	elif right_most_children:
		child_type = 2

	parent_type = 0
	if has_left_children and has_right_children:
		parent_type = 1
	elif has_right_children:
		parent_type = 2
	elif has_left_children:
		parent_type = 3

	return has_predicate_as_parent, child_type, parent_type

def get_child_absolution_position(tree):
	childpos = tree.treepositions("leaves")
	tree_pos = tree.treeposition()
	return [tree_pos + p[:-1] for p in childpos]

def get_constrain_seq(tree, predicate_index):
	list_subtree = [ch for ch in tree.subtrees(lambda t: t.height() == 2)]
	predicate = list_subtree[predicate_index]
	leaves_list = tree.treepositions('leaves')
	bio_ann = ['O'] * len(leaves_list)
	current_node = predicate
	while current_node is not tree.root():
		right_sibling = current_node.right_sibling()
		while right_sibling is not None:
			add = True
			conj = right_sibling.right_sibling() is not None and right_sibling.label() == 'CC' and right_sibling.right_sibling().label() == current_node.label()
			if conj:
				right_sibling = right_sibling.right_sibling()
				add = False
			elif right_sibling.label() in list_for_punctuation:
				add = False

			if add:
				if right_sibling.label().startswith('PP'):
					candidate = get_child_absolution_position(right_sibling )
					for j, tree_position in enumerate(leaves_list):
						if tree_position[:-1] == candidate[0]:
							for x in range(j, j + len(candidate)):
								bio_ann[x] = 'R-PP'
							break
				else:
					l = right_sibling.label() if right_sibling.height() > 2 else 'POS'
					candidate = get_child_absolution_position(right_sibling)
					for j, tree_position in enumerate(leaves_list):
						if tree_position[:-1] == candidate[0]:
							bio_ann[j] = 'B-%s' % l
							for x in range(j + 1, j + len(candidate)):
								bio_ann[x] = 'I-%s' % l
							break
			right_sibling = right_sibling.right_sibling()
		
		left_sibling = current_node.left_sibling()
		while left_sibling is not None:
			add = True
			conj = left_sibling.left_sibling() is not None and left_sibling.label() == 'CC' and left_sibling.left_sibling().label() == current_node.label()
			if conj:
				left_sibling = left_sibling.left_sibling()
				add = False
			elif left_sibling.label() in list_for_punctuation:
				add = False

			if add:
				if left_sibling.label().startswith('PP'):
					candidate = get_child_absolution_position(left_sibling)
					for j, tree_position in enumerate(leaves_list):
						if tree_position[:-1] == candidate[0]:
							for x in range(j, j + len(candidate)):
								bio_ann[x] = 'R-PP'
							break
				else:
					l = left_sibling.label() if left_sibling.height() > 2 else 'POS'
					candidate = get_child_absolution_position(left_sibling)
					for j, tree_position in enumerate(leaves_list):
						if tree_position[:-1] == candidate[0]:
							bio_ann[j] = 'B-%s' % l
							for x in range(j + 1, j + len(candidate)):
								bio_ann[x] = 'I-%s' % l
							break
			left_sibling = left_sibling.left_sibling()
		current_node = current_node.parent()

	return bio_ann

def read_parse_annotation(filename):
	mapping = {}
	mapping_feature = {}
	with open(filename) as parse:
		for line in parse:
			line = line.strip()
			ptree = ParentedTree.fromstring(line)
			text = ' '.join([process_word(t) for t in ptree.leaves()])
			trans_operation = get_trans_tree(line) if len(ptree.leaves()) > 1 else ['[']
			mapping[text] = (ptree, get_tree_encode_feature(line), trans_operation)

	return mapping

def get_trans_tree(tree_str):
	tree = string_trees(tree_str)
	tree = prune(tree[0], True, True, True, binlabelf=lambda _:'*')
	labels = tree_labels(tree)
	return [l[0] for l in labels]

def get_tree_encode_feature(tree_str):
	tree = SeqTree.fromstring(tree_str, remove_empty_top_bracketing=True)
	tree.set_encoding(RelativeLevelTreeEncoder())
	tree_labels = tree.to_maxincommon_sequence(root_label=True, encode_unary_leaf=False)
	label, relative_layer = [], []
	current_height = 0
	for tl in tree_labels:
		items = tl.split('_')
		if len(items) == 2:
			label.append(items[1])
			relative_layer.append(items[0])
		else:
			label.append('NONE')
			relative_layer.append('NONE')

	return label, relative_layer