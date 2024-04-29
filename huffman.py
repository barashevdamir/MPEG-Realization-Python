import collections
import heapq
from typing import List, Dict, Tuple

def huffman_encoding(data: List[Tuple[str, int]]) -> Dict[str, str]:
    """
    Построение таблицы кодирования Хаффмана из списка кортежей (value, count).
    """
    if not all(isinstance(item, tuple) and len(item) == 2 for item in data):
        raise ValueError("Input data must be a list of tuples (value, count)")

    frequency = collections.Counter(data)
    heap = [[freq, [symbol, ""]] for symbol, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p))
    return {item[0]: item[1] for item in huff}

def huffman_decoding(encoded_data: str, huffman_table: Dict[str, str]) -> List[str]:
    """
    Декодирование Хаффмана.
    """
    if not isinstance(encoded_data, str):
        raise ValueError("Encoded data must be a string")

    reverse_huffman_table = {v: k for k, v in huffman_table.items()}
    decoded_blocks = []
    current_code = []

    for bit in encoded_data:
        current_code.append(bit)
        current_code_str = ''.join(current_code)
        if current_code_str in reverse_huffman_table:
            decoded_blocks.append(reverse_huffman_table[current_code_str])
            current_code = []

    return decoded_blocks
