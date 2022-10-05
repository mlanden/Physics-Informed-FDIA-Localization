from collections import defaultdict
import numpy as np


class MISTree:
    def __init__(self):
        self.root = None
        self.table = MISTable()
        self.root = TreeNode("root", None, 0)

    def build(self, database, items_satisfied, min_supports):
        for transaction in database:
            item_list = []
            if type(transaction) is np.ndarray:
                transaction = tuple(transaction)

            for item in items_satisfied[transaction]:
                item_list.append((item, min_supports[item]))
            item_list.sort(key=lambda tup: tup[1], reverse=True)
            self._insert(item_list)

    def _insert(self, item_list):
        node = self.root
        for item, min_support in item_list:
            added = False
            for child in node.children:
                if child.item == item:
                    self.table.update_support(child)
                    child.count += 1
                    added = True
                    node = child
                    break

            if not added:
                child = TreeNode(item, node, 1)
                node.children.append(child)
                if item not in self.table:
                    self.table.add(item, min_support)
                self.table.update_support(child)
                node = child

    def build_conditional(self, prefix_paths, min_supports):
        for path_ in prefix_paths:
            node = self.root
            for item, count in path_:
                added = False
                for child in node.children:
                    if item == child.item:
                        self.table.update_support(child, count)
                        added = True
                        node.count += count
                        node = child
                        break

                if not added:
                    child = TreeNode(item, node, count)
                    node.children.append(child)
                    if item not in self.table:
                        self.table.add(item, min_supports[item])
                    self.table.update_support(child, count)
                    node = child

    def prune(self):
        self._prune()
        self._merge()
        self._infrequent_leaf_pune()

    def _prune(self):
        items = list(self.table.support.keys())
        items.sort(key=lambda x: (self.table.min_support[x], self.table.support[x]))

        least_min_support = 0
        for item in items:
            if self.table.support[item] < self.table.min_support[item]:
                self._mis_prune(item)
                del self.table[item]
            else:
                least_min_support = self.table.min_support[item]
                break

        for item in list(self.table.min_support):
            if self.table.min_support[item] < least_min_support:
                self._mis_prune(item)
                del self.table[item]

    def _mis_prune(self, item):
        for i in range(len(self.table.node_list[item])):
            node = self.table.node_list[item][i]
            parent = node.parent
            parent.children.remove(node)
            if len(node.children) > 0:
                parent.children.extend(node.children)
                for child in node.children:
                    child.parent = parent

    def _merge(self):
        nodes = [self.root]
        while len(nodes) > 0:
            items = {}
            node = nodes.pop()
            i = 0
            while i < len(node.children):
                child = node.children[i]
                if child.item in items:
                    first_node = items[child.item]
                    first_node.count += child.count
                    del node.children[i]
                    self.table.node_list[child.item].remove(child)
                else:
                    items[child.item] = child
                    nodes.append(child)
                    i += 1

    def _infrequent_leaf_pune(self):
        items = list(self.table.support.keys())
        items.sort(key=lambda x: (self.table.min_support[x], self.table.support[x]))

        for i in range(1, len(items)):
            item = items[i]
            if self.table.support[item] < self.table.min_support[item]:
                for node in self.table.node_list[item]:
                    if len(node.children) == 0:
                        node.parent.children.remove(node)
                        self.table.node_list[item].remove(node)
            if len(self.table.node_list[item]) == 0:
                del self.table[item]

    def conditional_prune(self, conditional_min_support):
        for item in dict(self.table.support):
            if self.table.support[item] < conditional_min_support:
                self._mis_prune(item)
                del self.table[item]
        self._merge()

    def support(self, pattern):
        items = []
        for item in pattern:
            items.append((item, self.table.min_support[item]))
        items.sort(key=lambda tup: tup[1])

        count = 0
        for start in self.table.node_list[items[0][0]]:
            i = 1
            parent = start.parent
            while parent.parent is not None and i < len(items):
                if parent.item == items[i][0]:
                    i += 1
                parent = parent.parent

            if i == len(items):
                count += 1

        return count

    def __str__(self):
        nodes = [(self.root, 0)]
        out = str(self.table)
        while len(nodes) > 0:
            node, tab = nodes.pop()
            out += '\t' * tab + node.item if node.item is not None else "Root"
            out += ":" + str(node.count) + ": ["
            for child in node.children:
                out += child.item + ", "
                if child is not None:
                    nodes.append((child, tab + 1))
            out += "]\n"
        return out


class TreeNode:
    def __init__(self, item, parent, count):
        self.children = []
        self.parent = parent
        self.item = item
        self.count = count

    def __str__(self):
        out = str(self.item)
        out += " Parent: "
        if self.parent is not None:
            out += str(self.parent.item)
        else:
            out += "Root"
        out += " Children: ["
        for child in self.children:
            out += str(child.item) + ", "
        out += "]"
        return out


class MISTable:
    def __init__(self):
        self.support = {}
        self.min_support = {}
        self.node_list = {}

    def add(self, item, mis_support):
        self.min_support[item] = mis_support
        self.support[item] = 0
        self.node_list[item] = []

    def update_support(self, node: TreeNode, count=1):
        self.support[node.item] += count
        if node.item not in self.node_list:
            self.node_list[node.item] = []
        if node not in self.node_list[node.item]:
            self.node_list[node.item].append(node)

    def __delitem__(self, item):
        if item in self.support:
            del self.support[item]
            del self.min_support[item]

        if item in self.node_list:
            del self.node_list[item]

    def __contains__(self, item):
        return item in self.min_support

    def __str__(self):
        out = "Item\t Support\t MIS\t Nodes\n"
        for i in self.support:
            out += i + "\t" + str(self.support[i]) + "\t" + str(self.min_support[i]) + "\t"
            for node in self.node_list[i]:
                out += str(node) + ", "
            out += "\n"
        return out


def cfp_growth(database, items_satisfied, min_supports, max_depth):
    tree = MISTree()
    tree.build(database, items_satisfied, min_supports)
    tree.prune()

    items = list(tree.table.support.keys())
    items.sort(key=lambda x: (tree.table.min_support[x], tree.table.support[x]))
    freq_patterns = []
    pattern_counts = defaultdict(lambda: 0)

    for item in items:
        cmin_support = tree.table.min_support[item]
        if tree.table.support[item] >= cmin_support:
            conditional_tree, conditional_pattern_bases = conditional_mis_tree(tree, item, [item], min_supports,
                                                                               cmin_support, -1)
            freq_patterns.extend(conditional_pattern_bases)
            for pattern in conditional_pattern_bases:
                pattern_counts[frozenset(pattern)] += 1

            if len(conditional_tree.table.support) > 0:
                cfp_growth_helper(conditional_tree, [item], cmin_support, min_supports, freq_patterns, pattern_counts,
                                  max_depth)

    return freq_patterns, pattern_counts, tree


def cfp_growth_helper(tree, pattern_base, cmin_support, min_supports, freq_patterns: list, pattern_counts: dict,
                      max_depth: int, depth=0):
    for item in tree.table.min_support:
        new_pattern_base = list(pattern_base)
        new_pattern_base.insert(0, item)
        conditional_tree, conditional_pattern_bases = conditional_mis_tree(tree, item, new_pattern_base, min_supports,
                                                                           cmin_support, depth)
        freq_patterns.extend(conditional_pattern_bases)
        for pattern in conditional_pattern_bases:
            pattern_counts[frozenset(pattern)] += 1

        if len(conditional_tree.table.support) > 0 and len(new_pattern_base) < max_depth:
            cfp_growth_helper(conditional_tree, new_pattern_base, cmin_support, min_supports, freq_patterns,
                              pattern_counts, max_depth, depth + 1)


def conditional_mis_tree(tree, item, pattern_base, min_supports, cmin_support, depth):
    pattern_bases = []
    for start_node in tree.table.node_list[item]:
        pattern = []
        min_support = start_node.count
        node = start_node.parent
        while node.parent is not None:
            pattern.insert(0, [node.item, min_support])
            node = node.parent
        pattern_bases.append(pattern)

    conditional_tree = MISTree()
    conditional_tree.build_conditional(pattern_bases, min_supports)
    conditional_tree.conditional_prune(cmin_support)

    conditional_pattern_bases = []
    for i in conditional_tree.table.support:
        if conditional_tree.table.support[i] >= cmin_support:
            pattern = list(pattern_base)
            pattern.insert(0, i)
            conditional_pattern_bases.append(pattern)

    return conditional_tree, conditional_pattern_bases


if __name__ == '__main__':
    example = 1
    if example == 0:
        db = list(range(20))
        satisfied = {0: ['a', 'b'], 1: ['a', 'e', 'f'], 2: ['c', 'd'], 3: ['a', 'b', 'h'], 4: ['c', 'd'],
                     5: ['a', 'c'],
                     6: ['a', 'b'], 7: ['e', 'f'], 8: ['c', 'd', 'g'], 9: ['a', 'b'], 10: ['a', 'b'],
                     11: ['a', 'c'],
                     12: ['a', 'b'], 13: ['b', 'e', 'f', 'g'], 14: ['c', 'd'], 15: ['a', 'b', 'd'],
                     16: ['c', 'd'],
                     17: ['a', 'c'], 18: ['a', 'b', 'e'], 19: ['c', 'd']}

        minimum_support = {'a': 10, 'b': 8, 'c': 10, 'd': 6, 'e': 3, 'f': 3, 'g': 3, 'h': 2}
    else:
        db = list(range(5))
        satisfied = {0: ['d', 'c', 'a', 'f'], 1: ['g', 'c', 'a', 'f', 'e'], 2: ['b', 'a', 'c', 'f', 'h'],
                     3: ['g', 'b', 'f'], 4: ['b', 'c']}
        minimum_support = {'a': 4, 'b': 4, 'c': 4, 'd': 3, 'e': 3, 'f': 2, 'g': 2, 'h': 2}

    frequent_patterns, pattern_counts = cfp_growth(db, satisfied, minimum_support, max_depth=4)
    print(frequent_patterns)
