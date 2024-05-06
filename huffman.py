import heapq


def build_huffman_tree(symbol_freq):
    """ Построение дерева Хаффмана из частот символов."""
    heap = [[wt, [sym, ""]] for sym, wt in symbol_freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p))

def huffman_encode(rle_encoded, tree):
    """ Кодирует список кортежей (значение, количество) по дереву Хаффмана, включая количество."""
    huffman_code = {sym: code for sym, code in tree}
    encoded = []
    for value, count in rle_encoded:
        # Добавляем кортеж из кода Хаффмана и количества повторений
        encoded.append((huffman_code[value], count))
    return encoded

def huffman_decode(encoded, tree):
    """
    Декодирует последовательность кодов Хаффмана в соответствии с деревом.
    Каждый код в списке `encoded` ассоциируется с количеством повторений,
    и функция возвращает список кортежей (значение, количество).
    """
    reverse_tree = {code: sym for sym, code in tree}
    decoded = []
    for code, count in encoded:
        decoded.append((reverse_tree[code], count))
    return decoded
