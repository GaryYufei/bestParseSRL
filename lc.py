"""
tree_labels() maps a binary tree to a label sequence

labels_tree() maps the label sequence back to the binarised tree

Here is an example:

Source tree: ['S', ['NP', ['NNP', 'Ms.'], ['NNP', 'Haag']], ['VP', ['VBZ', 'plays'], ['NP', ['NNP', 'Elianti']]], ['.', '.']]

Binarised tree: ['S', ['NP.VP', ['NP', ['NNP', 'Ms.'], ['NNP', 'Haag']], ['VP', ['VBZ', 'plays'], ['NP', 'Elianti']]], ['.', '.']]

Labels: [('[', 'S', 'Ms.'), ('><', 'NNP', 'NNP', 'Haag'), ('>[', 'NP', 'VP', 'plays'), ('><]', 'VBZ', 'NP', 'Elianti'), ('><]', 'NP.VP', '.', '.')]

Each element in Labels is a quadruple or a triple (only the first word only has 3 elements).

There is one element for each terminal (word) in the source tree.

The last element is the word, and the previous elements are the label
for the word.
"""

import tb

def make_pair(a,b):
    """Constructor for left-corner pair categories."""
    return a+'_'+b

def pair_categories(p):
    """Accessor for left-corner pair categories."""
    return p.split('_',1)

def is_pair(p):
    """True iff this is a pair category."""
    return '_' in p

def flag(tree):
    return [tree[0]+'+']+tree[1:]

def is_flagged(tree):
    if tree[0][-1] == '+':
        return tree[0][:-1]
    
def unflag(tree):
    return [tree[0][:-1]]+tree[1:]

        
def cat_a(S, a):
    return ('[',S,a[1])

def cat_b(A, X, B, a):
    return ('><',X,a[0],a[1])

def cat_c(A, X, a):
    return ('><]',X,a[0],a[1])

def cat_d(A, X, B, C, a):
    return ('>[',X,C,a[1])

def cat_e(A, X, C, a):
    """Note: we use '(' instead of '<<'"""
    return ('>(',X,C,a[1])

def lcx2(root):

    """
    lcx2() maps a binary tree into the left-corner transform of my 1996 paper.
    Preterminals that are right children, i.e., generated under schema (11b) and (11c),
    are flagged.  This permits us to distinguish schema (11b) and (11e).
    """

    def rightbranch(tree):
        
        """
        Transform a subtree lying on a right branch.
        """

        def leftbranch(subtree, transformed_right):
            """
            Transform a subtree lying on a left branch.
            transformed_right is transformed right material between this node and Anc.
            """
            if tb.is_preterminal(subtree):
                return [subtree, transformed_right]
            else:
                left, right = tb.tree_children(subtree)
                return leftbranch(left,
                                  tb.make_nonterminal(make_pair(Anc, tb.tree_label(left)),
                                                      rightbranch(right) + [transformed_right]))

        if tb.is_preterminal(tree):
            return [flag(tree)]
        else:
            Anc = tb.tree_label(tree)
            left, right = tb.tree_children(tree)
            return leftbranch(left,
                              tb.make_nonterminal(make_pair(Anc, tb.tree_label(left)),
                                                  rightbranch(right)))

    if tb.is_preterminal(root):
        return root
    else:
        return tb.make_nonterminal(tb.tree_label(root),
                                   rightbranch(root))
    
    
def lcx2tree_labels0(xtree):
    """
    Maps an lcx2 tree to the corresponding labels, as in my 1996 paper.
    """
    def visit(node, sofar):
        assert tb.is_phrasal(node)
        label = tb.tree_label(node)
        assert is_pair(label)
        A, X = pair_categories(label)
        children = tb.tree_children(node)
        assert len(children) > 0
        assert tb.is_preterminal(children[0])
        if len(children) == 1:
            assert is_flagged(children[0])
            xf = cat_c(A, X, unflag(children[0]))
        elif len(children) == 2:
            if is_flagged(children[0]):
                A1, B = pair_categories(tb.tree_label(children[1]))
                assert A1 == A
                xf = cat_b(A, X, B, unflag(children[0]))
            else:
                C, a = pair_categories(tb.tree_label(children[1]))
                assert a == tb.tree_label(children[0]), \
                       "error in label of node.children[1] a = {}, node = {}".format(a, node)
                xf = cat_e(A, X, C, children[0])
        elif len(children) == 3:
            assert not is_flagged(children[0])
            C, a = pair_categories(tb.tree_label(children[1]))
            A1, B = pair_categories(tb.tree_label(children[2]))
            assert A == A1
            xf = cat_d(A, X, B, C, children[0])
        else:
            sys.exit("error: ill-formed subtree {}\n in tree {}".format(node, xtree))
        sofar.append(xf)
        for child in children[1:]:
            sofar = visit(child, sofar)
        return sofar
    
    root = tb.tree_label(xtree)
    rchildren = tb.tree_children(xtree)
    assert len(rchildren) == 2, "nonbinary xtree = {}".format(xtree)
    sofar = [cat_a(root, rchildren[0])]
    return visit(rchildren[1], sofar)

# lcx2tree_labels0(lcx2(tb.prune(t0, True, True, True)))

def tree_labels(tree):
    return lcx2tree_labels0(lcx2(tree))

def labels_tree(labels):

    def visit(ls, subtree):
        L, X, B, w = ls[0]
        ls = ls[1:]
        if L == '><':
            return visit(ls, [[X]+subtree,[B,w]])
        elif L == '><]':
            return ls, [[X]+subtree,[B,w]]
        elif L == '>[':
            ls1, subtree1 = visit(ls, [w])
            return visit(ls1, [[X]+subtree,[B]+subtree1])
        elif L == '>(':
            ls1, subtree1 = visit(ls, [w])
            return ls1, [[X]+subtree,[B]+subtree1]
        assert False

    L0, S, w = labels[0]
    assert L0 == '['
    labels1, tree = visit(labels[1:], [w])
    assert labels1 == []
    return [S]+tree

###################

def lcx0(t):
    
    """
    lcx0() maps a binary tree into its left-corner transform as in my 1996 paper.
    """

    def _lcx(s, cont):
        if tb.is_preterminal(s):
            return [A,s]+cont
        else:
            return _lcx(s[1], [[make_pair(A,s[1][0]), lcx0(s[2])]+cont])
    
    if tb.is_preterminal(t):
        return t
    else:
        A = t[0]
        return _lcx(t, []) 

    
def lcx(root):

    """lcx() maps a binary tree into the left-corner transform of my 1996 paper."""

    def rightbranch(tree):

        def leftbranch(subtree, continuation):
            if tb.is_preterminal(subtree):
                return [subtree]+continuation
            else:
                left, right = tb.tree_children(subtree)
                return leftbranch(left,
                                  [tb.make_nonterminal(make_pair(Anc,tb.tree_label(left)),
                                                       rightbranch(right) + continuation)])

        Anc = tb.tree_label(tree)
        return leftbranch(tree, [])

    if tb.is_preterminal(root):
        return root
    else:
        return tb.make_nonterminal(tb.tree_label(root),
                                   rightbranch(root))


def lct(root):

    """
    lct() implements the same transform as lcx(), but it also relabels
    the preterminal labels to implement the transduction in my 1996
    paper.
    
    It isn't complete, i.e., it doesn't implement the relabelling.

    """

    def relabel(tree, label):
        return tb.make_nonterminal(tree[0]+' '+label, tb.tree_children(tree))

    def rightbranch(tree, X0):

        def leftbranch(subtree, continuation, X1):
            if tb.is_preterminal(subtree):
                return [relabel(subtree, X1)]+continuation
            else:
                left, right = tb.tree_children(subtree)
                X2 = tb.tree_label(left)+'>'
                return leftbranch(left,
                                  [tb.make_nonterminal(make_pair(Anc,tb.tree_label(left)),
                                                       rightbranch(right, X2)+continuation)],
                                  X1)

        if tb.is_preterminal(tree):
            return [relabel(tree, X0+'<'+tb.tree_label(tree)+']')]
        else:
            Anc = tb.tree_label(tree)
            left, right = tb.tree_children(tree)
            X2 = tb.tree_label(left)+'>'
            return leftbranch(left,
                              [tb.make_nonterminal(make_pair(Anc,tb.tree_label(left)),
                                                   rightbranch(right, X2))],
                              X0)

    if tb.is_preterminal(root):
        return root
    else:
        return tb.make_nonterminal(tb.tree_label(root),
                                   rightbranch(root, ''))
    
            

def make_binary_tree(depth, context=''):
    """this produces trees for testing the left-corner transform"""
    if depth <= 0:
        return tb.make_preterminal('n'+context, 'w'+context)
    else:
        return tb.make_nonterminal('n'+context,
                                   [make_binary_tree(depth-1, context+'1'),
                                    make_binary_tree(depth-1, context+'2')])

def make_leftbranching_tree(depth, context=''):
    """this produces trees for testing the left-corner transform"""
    if depth <= 0:
        return tb.make_preterminal('n'+context, 'w'+context)
    else:
        return tb.make_nonterminal('n'+context,
                                   [make_leftbranching_tree(depth-1, context+'1'),
                                    make_leftbranching_tree(0, context+'2')])

def make_rightbranching_tree(depth, context=''):
    """this produces trees for testing the left-corner transform"""
    if depth <= 0:
        return tb.make_preterminal('n'+context, 'w'+context)
    else:
        return tb.make_nonterminal('n'+context,
                                   [make_rightbranching_tree(0, context+'1'),
                                    make_rightbranching_tree(depth-1, context+'2')])


# from importlib import reload
# from drawtree import tb_tk
# import tb

# t0 = ['S',['NP',['DT','the'],['N1',['N1',['N','cheese']],['RC',['COMP','that'],['S/NP',['NP',['PN','Kim']],['VP/NP',['VT','likes'],['NP/NP']]]]]],['VP',['VT','disappointed'],['NP',['PN','Sandy']]]]

# tb_tk(lc.lcx(lc.make_binary_tree(4)))
# tb_tk(lc.lcx(tb.prune(lc.t0, True, True, True)))
