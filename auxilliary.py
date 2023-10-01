# Auxilliary code
import copy
from itertools import chain, combinations
from typing import Dict, Set

import numpy as np


class MCTSNode:
    def __init__(self, parent, children, Nc, value, belief_particles):
        self.parent = parent
        self.children = children
        self.visit_count = Nc  # Visit count of the node.
        self.value = value  # Estimated value of the node.
        self.belief_particles: Set = belief_particles  # A list that contains special data, or -1 if it's an action node.


# builds a tree
class BuildTree:
    def __init__(
        self, root_parameters=MCTSNode("isRoot", {}, 0, 0, set())
    ):  # add init dist
        # index for nodes
        self.index_for_node = -1
        self.nodes: Dict[int, MCTSNode] = {}
        # dictionary where key is node and value is list of corresponding values
        # = [ parent, children, Nc, Value, B() (-1) if action node]

        self.root_parameters = root_parameters
        self.nodes[self.index_for_node] = copy.deepcopy(self.root_parameters)

    # Expand the tree by one node.
    # If the result of an action give IsAction = True
    def expand_tree_by_one_node(self, parent, action_or_observation, is_action=False):
        self.index_for_node += 1
        if is_action:
            # add node to tree
            self.nodes[self.index_for_node] = MCTSNode(parent, {}, 0, 0, -1)
            # inform parent node
            self.nodes[parent].children[action_or_observation] = self.index_for_node
        else:
            self.nodes[self.index_for_node] = MCTSNode(parent, {}, 0, 0, set())
            self.nodes[parent].children[action_or_observation] = self.index_for_node

    # Check given nodeindex corresponds to leaf node
    def is_leaf_node(self, n):
        if self.nodes[n].visit_count == 0:
            return True
        else:
            return False

    # As in POMCP/ Checks that an observation was already made before moving
    def get_node_from_observation(self, base_node, sample_observation):
        # Check if a given observation node has been visited
        if sample_observation not in list(self.nodes[base_node].children.keys()):
            # If not create the node
            self.expand_tree_by_one_node(base_node, sample_observation)
        # Get the nodes index
        next_node = self.nodes[base_node].children[sample_observation]
        return next_node

    # Removes a node and all its children
    def prune(self, node):
        children = self.nodes[node].children
        del self.nodes[node]
        for _, child in children.items():
            self.prune(child)

    # make new root and update children
    def make_new_root(self, new_root):
        self.nodes[-1] = copy.copy(self.nodes[new_root])
        del self.nodes[new_root]
        self.nodes[-1].parent = "isRoot"
        # update children
        for _, child in self.nodes[-1].children.items():
            self.nodes[child].parent = -1

    # Prune tree after action and observation were made
    def prune_after_action(self, action, observation):
        # Get node after action
        action_node = self.nodes[-1].children[action]

        # Get new root (after observation)
        new_root = self.get_node_from_observation(action_node, observation)

        # remove new_root from parent's children to avoid deletion
        del self.nodes[action_node].children[observation]

        # prune unnesecary nodes
        self.prune(-1)

        # set new_root as root (key = -1)
        self.make_new_root(new_root)


# UCB score calculation
def UCB(N, n, V, c=1):  # N=Total, n= local, V = value, c = parameter
    return V + c * np.sqrt(np.log(N) / n)


# from itertools recipes
# creates power set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
