"""tb.py reads, searches and displays trees from Penn Treebank (PTB) format
treebank files.

Mark Johnson, 14th January, 2012, last modified 19th January 2016

Trees are represented in Python as nested list structures in the following
format:

  Terminal nodes are represented by strings.

  Nonterminal nodes are represented by lists.  The first element of
  the list is the node's label (a string), and the remaining elements
  of the list are lists representing the node's children.

This module also defines two regular expressions.

nonterm_rex matches Penn treebank nonterminal labels, and parses them into
their various parts.

empty_re matches empty elements (terminals), and parses them into their
various parts.
"""

import collections, glob, re, sys

_header_re = re.compile(r"(\*x\*.*\*x\*[ \t]*\n)*\s*")
_openpar_re = re.compile(r"\s*\(\s*([^ \t\n\r\f\v()]*)\s*")
_closepar_re = re.compile(r"\s*\)\s*")
_terminal_re = re.compile(r"\s*([^ \t\n\r\f\v()]*)\s*")

# This is such a complicated regular expression that I use the special
# "verbose" form of regular expressions, which lets me index and document it
#
nonterm_rex = re.compile(r"""
^(?P<CAT>[A-Z0-9$|^]+)                                  # category comes first
 (?:                                                    # huge disjunct of optional annotations
     - (?:(?P<FORMFUN>ADV|NOM)                          # stuff beginning with -
        |(?P<GROLE>DTV|LGS|PRD|PUT|SBJ|TPC|VOC)
        |(?P<ADV>BNF|DIR|EXT|LOC|MNR|PRP|TMP)
        |(?P<MISC>CLR|CLF|HLN|SEZ|TTL)
        |(?P<TPC>TPC)
        |(?P<DYS>UNF|ETC|IMP)
        |(?P<INDEX>[0-9]+)
       )
  | = (?P<EQINDEX>[0-9]+)                               # stuff beginning with =
 )*                                                     # Kleene star
$""", re.VERBOSE)

empty_re = re.compile(r"^(?P<CAT>[A-Z0-9\?\*]+)(?:-(?P<INDEX>\d+))")

def read_file(filename):

    """Returns the trees in the PTB file filename."""
    
    filecontents = open(filename, "rU").read()
    pos = _header_re.match(filecontents).end()
    trees = []
    _string_trees(trees, filecontents, pos)
    return trees

def string_trees(s):
    
    """Returns a list of the trees in PTB-format string s"""
    
    trees = []
    _string_trees(trees, s)
    return trees

def _string_trees(trees, s, pos=0):
    
    """Reads a sequence of trees in string s[pos:].
    Appends the trees to the argument trees.
    Returns the ending position of those trees in s."""
    
    while pos < len(s):
        closepar_mo = _closepar_re.match(s, pos)
        if closepar_mo:
            return closepar_mo.end()
        openpar_mo = _openpar_re.match(s, pos)
        if openpar_mo:
            tree = [openpar_mo.group(1)]
            trees.append(tree)
            pos = _string_trees(tree, s, openpar_mo.end())
        else:
            terminal_mo = _terminal_re.match(s, pos)
            trees.append(terminal_mo.group(1))
            pos = terminal_mo.end()
    return pos


def make_nonterminal(label, children):
    
    """returns a tree node with root node label and children"""

    return [label]+children


def make_terminal(word):

    """returns a terminal tree node with label word"""

    return word

def make_preterminal(label, word):

    """returns a preterminal node with label for word"""

    return [label, word]


def is_terminal(subtree):
    
    """True if this subtree consists of a single terminal node
    (i.e., a word or an empty node)."""
    
    return not isinstance(subtree, list)


def is_nonterminal(subtree):
    
    """True if this subtree does not consist of a single terminal node
    (i.e., a word or an empty node)."""
    
    return isinstance(subtree, list)


def is_preterminal(subtree):
    
    """True if the treebank subtree is rooted in a preterminal node
    (i.e., is an empty node or dominates a word)."""
    
    return isinstance(subtree, list) and len(subtree) == 2 and is_terminal(subtree[1])


def is_phrasal(subtree):
    
    """True if this treebank subtree is not a terminal or a preterminal node."""
    
    return isinstance(subtree, list) and \
           (len(subtree) == 1 or isinstance(subtree[1], list))


def is_punctuation(subtree):

    """True if this subtree is a preterminal node dominating a punctuation or 
    empty node."""

    return is_preterminal(subtree) and \
        tree_category(subtree) in ("''",":","#",",",".","``","-LRB-","-RRB-","-NONE-")


def tree_children(tree):

    """Returns the children subtrees of tree"""

    if isinstance(tree, list):
        return tree[1:]
    else:
        return []


def tree_label(tree):

    """Returns the label on the root node of tree."""

    if isinstance(tree, list):
        return tree[0]
    else:
        return tree


def label_category(label):

    """Returns the category part of a label."""

    nonterm_mo = nonterm_rex.match(label)
    if nonterm_mo:
        return nonterm_mo.group('CAT')
    else:
        return label


def tree_category(tree):

    """Returns the category of the root node of tree."""

    if isinstance(tree, list):
        return label_category(tree[0])
    else:
        return tree


def map_labels(tree, fn):
    
    """Returns a tree in which every node's label is mapped by fn"""

    if isinstance(tree, list):
        return [fn(tree[0])]+[map_labels(child,fn) for child in tree[1:]]
    else:
        return tree


def label_noindices(label):
    
    """Removes indices in label if present"""

    label_mo = nonterm_rex.match(label)
    if label_mo:
        start = max(label_mo.end('INDEX'), label_mo.end('EQINDEX'))
        if start > 1:
            return label[:start-2]
    return label


def tree_children(tree):

    """Returns a list of the subtrees of tree."""

    if isinstance(tree, list):
        return tree[1:]
    else:
        return []


def tree_copy(tree):

    """Returns a deep copy of tree"""

    if isinstance(tree, list):
        return [tree_copy(child) for child in tree]
    else:
        return tree


def prune(tree, remove_empty=False, collapse_unary=False, binarise=False, 
          relabel=lambda x: x,
          binlabelf=lambda labels: '+'.join(labels)):

    """Returns a copy of tree without empty nodes, unary nodes or node indices.

    If binarise=='right' then right-binarise nodes, otherwise 
    if binarise is not False then left-binarise nodes.

    binlabelf() maps a sequence of child node labels to the label for 
    a new binarised node.
    """

    def left_binarise(cs, rightpos):
        label = binlabelf(tree_label(cs[i]) for i in range(rightpos))
        if rightpos <= 2:
            return make_nonterminal(label, cs[:rightpos])
        else:
            return make_nonterminal(label, [left_binarise(cs, rightpos-1),cs[rightpos-1]])

    def right_binarise(cs, leftpos, len_cs):
        label = binlabelf(tree_label(c) for c in cs[leftpos:])
        if leftpos + 2 >= len_cs:
            return make_nonterminal(label, cs[leftpos:])
        else:
            return make_nonterminal(label, [cs[leftpos], right_binarise(cs, leftpos+1, len_cs)])

    label = tree_label(tree)
    if is_phrasal(tree):
        cs = (prune(c, remove_empty, collapse_unary, binarise, relabel, binlabelf) 
              for c in tree_children(tree))
        cs = [c for c in cs if c]
        if cs or not remove_empty:
            len_cs = len(cs)
            if collapse_unary and len_cs == 1:
                return make_nonterminal(relabel(label), 
                                        tree_children(cs[0]))
            elif binarise and len_cs > 2:
                if binarise=='right':
                    return make_nonterminal(relabel(label),
                                            [cs[0], right_binarise(cs, 1, len_cs)])
                else:
                    return make_nonterminal(relabel(label),
                                            [left_binarise(cs, len_cs-1), cs[-1]])
            else:
                return make_nonterminal(relabel(label), 
                                        cs)
        else:
            return None
    elif is_preterminal(tree):
        if remove_empty and label == '-NONE-':
            return None
        else:
            return make_nonterminal(relabel(label), tree_children(tree))
    else:
        return tree


def tree_nodes(tree):
    
    """Yields the nodes in tree"""

    def visit(node):
        yield node
        if isinstance(node, list):
            for child in node[1:]:
                yield from visit(child)

    yield from visit(tree)
    

def tree_terminals(tree):
    
    """Yields the terminal leaves of tree"""

    def visit(node):
        if isinstance(node, list):
            for child in node[1:]:
                yield from visit(child)
        else:
            yield node

    yield from visit(tree)


def tree_preterminals(tree):

    """Yields the preterminal nodes of tree"""

    def visit(node):
        if is_preterminal(node):
            yield node
        else:
            for child in node[1:]:
                yield from visit(child)

    yield from visit(tree)


def tree_phrasalnodes(tree):

    """Yields the phrasal (i.e., nonterminal and non-preterminal) nodes of tree"""

    def visit(node):
        if is_phrasal(node):
            yield node
            for child in node[1:]:
                yield from visit(child)

    yield from visit(tree)


def tree_constituents(tree, collect_root=False, collect_terminals=False, 
                      collect_preterminals=False, ignore_punctuation=False):

    """maps a tree to a list of tuples (category,left,right) that
    correspond to constituents of the tree.

    If collect_root==True, then the list of tuples includes a tuple
    for the root node of the tree.

    If collect_terminals==True, then the list of tuples includes tuples
    for the terminal nodes of the tree.

    If collect_preterminals==True, then the list of tuples includes tuples
    for the preterminal nodes of the tree.

    If ignore_punctuation==True, then the left and right positions ignore
    punctuation.
    """

    def visitor(node, left, constituents):
        if ignore_punctuation and is_punctuation(node):
            return left
        if is_terminal(node):
            if collect_terminals:
                constituents.append((tree_category(node),left,left+1))
            return left+1
        else:
            right = left
            for child in tree_subtrees(node):
                right = visitor(child, right, constituents)
            if collect_preterminals or is_phrasal(node):
                constituents.append((tree_category(node),left,right))
            return right

    constituents = []
    if collect_root:
        visitor(tree, 0, constituents)
    else:
        right = 0
        for child in tree_subtrees(tree):
            right = visitor(child, right, constituents)
    return constituents


def write(tree, outf=sys.stdout):
    """Write a tree to outf"""
    if is_nonterminal(tree):
        outf.write('(')
        for i in range(0,len(tree)):
            if i > 0:
                outf.write(' ')
            write(tree[i], outf)
        outf.write(')')
    else:
        outf.write(tree)



def read_ptb(basedir="/usr/local/data/LDC/LDC2015T13_eng_news_txt_tbnk-ptb_revised/",
             remove_empty=True, collapse_unary=False, binarise=False, relabel=label_category):
    
    """Returns a tuple (train,dev,test) of the trees in 2015 PTB.  train, dev and test are generators
    that enumerate the trees in each section"""

    def _read_ptb(dirs):
        for p in dirs:
            for fname in sorted(glob.glob(basedir+p)):
                for tree in read_file(fname):
                    yield prune(tree[1], remove_empty, collapse_unary, binarise, relabel)

    ptb = collections.namedtuple('ptb', 'train dev test')
    return ptb(train=_read_ptb(("data/penntree/0[2-9]/wsj*.tree",
                                "data/penntree/1[2-9]/wsj*.tree",
                                "data/penntree/2[01]/wsj*.tree")),
               dev=_read_ptb(("data/penntree/24/wsj*.tree",)),
               test=_read_ptb(("data/penntree/23/wsj*.tree",)))

