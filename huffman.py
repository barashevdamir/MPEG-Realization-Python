import heapq
import collections


def encode(frequency: dict) -> dict:
    """
    Построение таблицы кодирования Хаффмана из списка кортежей (value, count).

    :param data: Список кортежей (value, count).
    :return: Словарь (value, code).

    """
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_encoding(data: list) -> dict:
    """
    Построение таблицы кодирования Хаффмана из списка кортежей (value, count).

    :param data: Список кортежей (value, count).
    :return: Словарь (value, code).

    """
    # Подсчет частоты вхождения каждого кортежа
    frequency = collections.Counter(data)
    # Создание мин-кучи с элементами в виде [вес, [символ, код]]
    heap = []
    for symbol, freq in frequency.items():
        heapq.heappush(heap, [freq, [symbol, ""]])

    # Строим дерево Хаффмана
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Конечный словарь кодов Хаффмана
    if heap:
        huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p))
        return {item[0]: item[1] for item in huff}
    else:
        return {}


def huffman_decoding(encoded_data: str, huffman_table: dict) -> list:
    """
    Декодирование Хаффмана
    :param encoded_data: закодированный блок
    :param huffman_table: таблица Хаффмана
    :return: декодированный блок
    """
    reverse_huffman_table = {v: k for k, v in huffman_table.items()}
    decoded_blocks = []
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huffman_table:
            decoded_blocks.append(reverse_huffman_table[current_code])
            current_code = ""
    return decoded_blocks


