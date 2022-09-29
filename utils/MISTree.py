

class MISTree:
    def __init__(self):
        self.root = None
        self.table = MISTable()

    def build(self, database, items_satisfied, min_supports):
        self.root = TreeNode(None, 0, [], [])
        for transaction in database:
            item_list = []
            transaction = tuple(transaction)
            for item in items_satisfied[transaction]:
                item_list.append((item, min_supports[item]))
            item_list.sort(key=lambda item, minsup: minsup, reverse=True)
            self._insert(item_list, min_supports)

    def _insert(self, item_list, min_supports):
        node = self.root
        for item in item_list:
            added = False
            for child in node.children:
                if child.item == item:
                    child.count += 1
                    added = True
                    node = child

            if not added:
                child = TreeNode(item, 1, node, [])
                node.children.append(child)

                if item not in self.table:
                    self.table.add(item, min_supports[item])

                self.table.update_support(item)


class TreeNode:
    def __init__(self, item, count, parent, children):
        self.children = children
        self.parent = parent
        self.count = count
        self.item = item


class MISTable:
    def __init__(self):
        self.support = {}
        self.min_support = {}

    def add(self, item, mis_support):
        self.min_support[item] = mis_support
        self.support[item] = 0

    def update_support(self, item):
        self.support[item] += 1

    def __contains__(self, item):
        return item in self.min_support