# -*- coding: utf-8 -*-
"""
Clement Michard (c) 2015
Ettore Forigo (c) 2020
"""

from collections import defaultdict
from pptree.utils import *
from typing import Dict


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


def get_tree_str_for_current_node(
    current_node, childattr="children", nameattr="name", horizontal=True
):
    if hasattr(current_node, nameattr):
        name = lambda node: getattr(node, nameattr)
    else:
        name = lambda node: str(node)

    children = lambda node: getattr(node, childattr)
    nb_children = lambda node: sum(nb_children(child) for child in children(node)) + 1

    def balanced_branches(current_node):
        size_branch = {child: nb_children(child) for child in children(current_node)}

        """ Creation of balanced lists for "a" branch and "b" branch. """
        a = sorted(children(current_node), key=lambda node: nb_children(node))
        b = []
        while a and sum(size_branch[node] for node in b) < sum(
            size_branch[node] for node in a
        ):
            b.append(a.pop())

        return a, b

    if horizontal:
        return get_tree_horizontally(current_node, balanced_branches, name)

    else:
        return get_tree_vertically(current_node, balanced_branches, name, children)


def get_tree_horizontally(
    current_node, balanced_branches, name_getter, indent="", last="updown"
):
    lines = []

    up, down = balanced_branches(current_node)

    """ Printing of "up" branch. """
    for child in up:
        next_last = "up" if up.index(child) == 0 else ""
        next_indent = "{0}{1}{2}".format(
            indent, " " if "up" in last else "│", " " * len(name_getter(current_node))
        )
        lines.append(
            get_tree_horizontally(
                child, balanced_branches, name_getter, next_indent, next_last
            )
        )

    """ Printing of current node. """
    if last == "up":
        start_shape = "┌"
    elif last == "down":
        start_shape = "└"
    elif last == "updown":
        start_shape = " "
    else:
        start_shape = "├"

    if up:
        end_shape = "┤"
    elif down:
        end_shape = "┐"
    else:
        end_shape = ""

    lines.append(
        "{0}{1}{2}{3}".format(indent, start_shape, name_getter(current_node), end_shape)
    )

    """ Printing of "down" branch. """
    for child in down:
        next_last = "down" if down.index(child) is len(down) - 1 else ""
        next_indent = "{0}{1}{2}".format(
            indent, " " if "down" in last else "│", " " * len(name_getter(current_node))
        )
        lines.append(
            get_tree_horizontally(
                child, balanced_branches, name_getter, next_indent, next_last
            )
        )
    return "\n".join(lines)


def tree_repr(current_node, balanced_branches, name, children):

    sx, dx = balanced_branches(current_node)

    """ Creation of children representation """

    tr_rpr = lambda node: tree_repr(node, balanced_branches, name, children)

    left = branch_left(map(tr_rpr, sx)) if sx else ()
    right = branch_right(map(tr_rpr, dx)) if dx else ()

    children_repr = tuple(connect_branches(left, right) if sx or dx else ())

    current_name = name(current_node)

    name_len = len(current_name)
    name_l, name_r = name_len // 2, name_len // 2

    left_len, right_len = blocklen(left), blocklen(right)

    current_name = (
        f"{' ' * (left_len - name_l)}{current_name}{' ' * (right_len - name_r)}"
    )

    return multijoin([[current_name, *children_repr]]), (
        max(left_len, name_l),
        max(right_len, name_r),
    )


def get_tree_vertically(*args):
    tree_repr(*args)[0]


def get_tree_str(child2parent: Dict, node2count: Dict):
    """treeをcountとともに出力する

    Args:
        child2parent (Dict): 階層構造を示す。子ノードから親ノードへのdict。ただしルートノードは"0"のvalをもつ
        node2count (Dict): ノードの各カウントを示す。比率を表示する場合はルートノードの頻度に対する比率を同時に表示する。
    """
    pass
    root = [c for c, p in child2parent.items() if p == 0]
    assert len(root) == 1
    root = root[0]
    root_count = node2count[root]
    name2node = {root: Node("%.2f %%,  %d, %s" % (100, root_count, root))}
    for node, count in node2count.items():
        pass
        name2node[node] = Node(
            "%.2f %%,  %d, %s" % (100 * count / root_count, count, node)
        )
    for child, parent in child2parent.items():
        if parent != 0:
            name2node[parent].add_child(name2node[child])
    return get_tree_str_for_current_node(name2node[root])


def make_node2count_consistently_with_child2parent(
    child2parent: Dict, node2count: Dict
):
    parent2children = defaultdict(set)
    for child, parent in child2parent.items():
        parent2children[parent].add(child)

    def top_node_to_sum_count(top_node: str):
        node_to_sum_count = dict()
        root_sum_count = node2count[top_node]
        for child in parent2children[top_node]:
            node_to_sum_count.update(top_node_to_sum_count(child))
            root_sum_count += node_to_sum_count[child]
        node_to_sum_count[top_node] = root_sum_count
        return node_to_sum_count

    root_node = [child for child, parent in child2parent.items() if parent == 0][0]
    return top_node_to_sum_count(root_node)
