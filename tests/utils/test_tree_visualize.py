import unittest
from src.utils.tree_visualize import (
    get_tree_str,
    make_node2count_consistently_with_child2parent,
)


class TestTreeVisualize(unittest.TestCase):
    def test_get_tree_str(self):
        child2parent = {
            "Person": "Animal",
            "Animal": "Creature",
            "Plant": "Creature",
            "Wood": "Plant",
            "Creature": 0,
        }
        node2count = {
            "Person": 1,
            "Animal": 20,
            "Plant": 30,
            "Wood": 10,
            "Creature": 100,
        }
        print(get_tree_str(child2parent, node2count))
        expected = """                         ┌20.00 %,  20, Animal┐
                         │                    └1.00 %,  1, Person
 100.00 %,  100, Creature┤
                         └30.00 %,  30, Plant┐
                                             └10.00 %,  10, Wood"""
        factual = get_tree_str(child2parent, node2count)
        for i, (e, f) in enumerate(zip(expected, factual)):
            if e != f:
                pass
        assert expected == factual
        pass

    def test_make_node2count_consistently_with_child2parent(self):
        child2parent = {
            "Person": "Animal",
            "Animal": "Creature",
            "Plant": "Creature",
            "Wood": "Plant",
            "Creature": 0,
        }
        node2count = {
            "Person": 1,
            "Animal": 19,
            "Plant": 20,
            "Wood": 10,
            "Creature": 50,
        }
        expected = {
            "Person": 1,
            "Animal": 20,
            "Plant": 30,
            "Wood": 10,
            "Creature": 100,
        }
        factual = make_node2count_consistently_with_child2parent(
            child2parent, node2count
        )
        assert expected == factual
