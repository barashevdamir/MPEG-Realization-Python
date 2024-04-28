import collections

NYT = "NYT"

class HuffmanNode:
    def __init__(self, symbol=None, weight=0):
        self.symbol = symbol
        self.weight = weight
        self.left = None
        self.right = None
        self.parent = None
        self.code = ""

    def is_leaf(self):
        return not self.left and not self.right

    def update_code(self, prefix=""):
        self.code = prefix
        if self.left:
            self.left.update_code(prefix + "0")
        if self.right:
            self.right.update_code(prefix + "1")


class AdaptiveHuffmanTree:
    def __init__(self):
        self.root = HuffmanNode(symbol=NYT)
        self.nodes = {NYT: self.root}
        self.next_number = 0

    def insert(self, symbol):
        if symbol in self.nodes:
            node = self.nodes[symbol]
        else:
            # New symbol, split NYT
            nyt = self.nodes[NYT]
            new_nyt = HuffmanNode(symbol=NYT, weight=0)
            new_leaf = HuffmanNode(symbol=symbol, weight=1)

            nyt.left = new_nyt
            nyt.right = new_leaf
            nyt.symbol = None  # NYT should not have a symbol anymore
            new_nyt.parent = nyt
            new_leaf.parent = nyt

            self.nodes[NYT] = new_nyt
            self.nodes[symbol] = new_leaf
            node = new_leaf

            self.update_tree(new_leaf)

        return self.get_code(node)

    def update_tree(self, node):
        while node:
            node.weight += 1
            node = node.parent
        self.root.update_code()

    def get_code(self, node):
        return node.code

    def encode(self, data):
        result = {}
        for symbol, count in data:
            for _ in range(count):
                code = self.insert(symbol)
                result[(symbol, count)] = code
        return result

    def get_huffman_table(self):
        huffman_table = {}
        self._build_table(self.root, "", huffman_table)
        return huffman_table

    def _build_table(self, node, code, table):
        if node.is_leaf():
            table[node.symbol] = code
        else:
            if node.left:
                self._build_table(node.left, code + '0', table)
            if node.right:
                self._build_table(node.right, code + '1', table)

# # Пример использования
# tree = AdaptiveHuffmanTree()
# data = [[(1.0, 1), (2.0, 1), (1.0, 1), (3.0, 1)], [(4.0, 1), (5.0, 1), (1.0, 1)]]
# encoded_data = tree.encode(data)
# print("Encoded:", encoded_data)
