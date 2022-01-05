class TreeNode:
    def __init__(self, info, is_leaf=False):
        self.info = info
        self.children = {}
        self.is_leaf = is_leaf

    def add(self, val):
        print(val)
        if len(val) == 1:
            if val in self.children:
                self.children[val].is_leaf = True
            else:
                self.children[val] = TreeNode(self.info + val, is_leaf=True)
            return
        if val[:1] in self.children:
            print(f"Found {val}")
            self.children[val[:1]].add(val[1:])
        else:
            self.children[val[:1]] = TreeNode(self.info + val[:1], is_leaf=False)
            self.children[val[:1]].add(val[:1])

    def search(self, query):
        print(query)
        if str(query[:1]) not in self.children:
            return False
        if len(query) == 1:
            return True
        return self.children[str(query[:1])].search(query[1:])

    def __repr__(self):
        return self.info + str([child.__repr__() for child in self.children.values()])


root = TreeNode("")
root.add("Tricche")
root.add("Tracche")
print(root)
print(root.search("Tricche"))
print(root.search("no"))
