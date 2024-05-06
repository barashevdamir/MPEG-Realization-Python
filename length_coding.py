def run_length_encode(block):
    """ Применяет RLE для массива."""
    flat_block = block.flatten()
    result = []
    last = flat_block[0]
    count = 1
    for i in flat_block[1:]:
        if i == last:
            count += 1
        else:
            result.append((last, count))
            last = i
            count = 1
    result.append((last, count))
    return result

def run_length_decode(encoded):
    """ Применяет обратное RLE для списка кортежей (значение, повторения)."""
    result = []
    for value, count in encoded:
        result.extend([value] * count)
    return result