import struct
import io
import os

# Константы размеров
BOX_HEADER_SIZE = 8
UUID_SIZE = 16


def get_language_string(language):
    """Преобразует числовое представление языка в строку по стандарту ISO-639-2/T."""
    lang = [(language >> 10) & 0x1F, (language >> 5) & 0x1F, language & 0x1F]
    return ''.join(chr(lang_part + 0x60) for lang_part in lang)


class Box:
    """Базовый класс для обработки данных бокса в формате MP4."""

    def __init__(self, name, size, start, reader):
        self.name = name
        self.size = size
        self.start = start
        self.reader = reader

    def read_box_data(self):
        """Читает и возвращает данные бокса, исключая заголовок."""
        if self.size <= BOX_HEADER_SIZE:
            return None
        return self.reader.read_bytes_at(self.size - BOX_HEADER_SIZE, self.start + BOX_HEADER_SIZE)

    def write_box_data(self, output):
        # Пересчитываем размер бокса на основе его содержимого
        self.size = BOX_HEADER_SIZE + len(self.content)
        # Записываем заголовок бокса
        output.write(struct.pack(">I4s", self.size, self.name.encode('utf-8')))
        # Записываем содержимое бокса
        output.write(self.content)

    def add_content(self, content):
        # Метод для добавления содержимого в бокс
        self.content += content


class Mp4Reader:
    """Основной класс для чтения и анализа структуры MP4 файла."""
    def __init__(self, reader):
        self.reader = reader
        self.ftyp = None
        self.moov = None
        self.mdat = None
        self.uuids = []
        self.size = 0
        self.is_fragmented = False

    def parse(self):
        """Анализирует структуру MP4 файла, идентифицируя основные боксы."""
        if self.size == 0:
            if isinstance(self.reader, io.FileIO):
                info = os.fstat(self.reader.fileno())
                self.size = info.st_size

        boxes = self.read_boxes(0, self.size)
        for box in boxes:
            if box.name == "ftyp":
                self.ftyp = FtypBox(box)
                self.ftyp.parse()
            elif box.name == "mdat":
                self.mdat = MdatBox(box)
            elif box.name == "moov":
                self.moov = MoovBox(box)
                self.moov.parse()
                self.is_fragmented = self.moov.is_fragmented
            elif box.name == "uuid":
                uuid_box = UuidBox(box)
                uuid_box.parse()
                self.uuids.append(uuid_box)

    def read_box_at(self, offset):
        """Читает заголовок бокса, возвращая его размер и тип."""
        buf = self.read_bytes_at(BOX_HEADER_SIZE, offset)
        if len(buf) < BOX_HEADER_SIZE:
            return 0, ""
        box_size = struct.unpack(">I", buf[0:4])[0]
        if offset + box_size > self.size:
            return 0, ""
        box_type = buf[4:8].decode("utf-8")
        return box_size, box_type

    def read_bytes_at(self, n, offset):
        """
        Читает заданное количество байтов из файла, начиная с указанного смещения.
        В случае ошибки чтения генерирует исключение IOError.
        """
        try:
            self.reader.seek(offset)
            buf = self.reader.read(n)
        except Exception as e:
            raise IOError(f"Error reading {n} bytes at offset {offset}: {str(e)}")

        if len(buf) < n:
            raise IOError(f"Could not read the specified number of bytes ({n}) from the file.")

        return buf

    def read_boxes(self, start, size):
        """Читает и анализирует боксы в указанном диапазоне."""
        boxes = []
        offset = start
        while offset < start + size:
            box_size, box_type = self.read_box_at(offset)
            if box_size == 0 or not box_type:
                break
            box = Box(box_type, box_size, offset, self)
            boxes.append(box)
            offset += box_size
        return boxes


class Mp4Writer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.boxes = []

    def add_box(self, box):
        """Добавляет бокс для записи."""
        self.boxes.append(box)

    def write(self):
        """Записывает MP4 файл с добавленными боксами."""
        with open(self.file_path, 'wb') as f:
            for box in self.boxes:
                box_data = box.write_box_data()
                f.write(box_data)


class Avc1Box(Box):
    """Обрабатывает 'avc1' бокс, содержащий информацию о видео-кодеке."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.avcC = None  # Конфигурация кодека

    def parse(self):
        """Извлекает данные о версии кодека и конфигурации."""
        data = self.read_box_data()
        self.version = data[0]


class EdtsBox(Box):
    """Обрабатывает 'edts' бокс, который содержит редакционную информацию о треке."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.elst = None

    def parse(self):
        """Анализирует вложенные боксы, находя и обрабатывая 'elst' бокс."""
        boxes = self.reader.read_boxes(self.start + BOX_HEADER_SIZE, self.size - BOX_HEADER_SIZE)
        for box in boxes:
            if box.name == "elst":
                self.elst = ElstBox(box)
                self.elst.parse()


class ElstBox(Box):
    """Обрабатывает 'elst' бокс, содержащий таблицу сегментации времени трека."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.entry_count = 0
        self.entries = []

    def parse(self):
        """Извлекает данные таблицы сегментации времени из бокса."""
        data = self.read_box_data()
        self.version = struct.unpack(">I", data[0:4])[0]
        self.entry_count = struct.unpack(">I", data[4:8])[0]
        offset = 8
        for i in range(self.entry_count):
            segment_duration = struct.unpack(">I", data[offset:offset+4])[0]
            offset += 4
            media_time = struct.unpack(">I", data[offset:offset+4])[0]
            offset += 4
            media_rate_integer = struct.unpack(">H", data[offset:offset+2])[0]
            offset += 2
            media_rate_fraction = struct.unpack(">H", data[offset:offset+2])[0]
            offset += 2
            entry = ElstEntry(segment_duration, media_time, media_rate_integer, media_rate_fraction)
            self.entries.append(entry)


class ElstEntry:
    """Представляет одну запись в таблице сегментации времени 'elst' бокса."""
    def __init__(self, segment_duration, media_time, media_rate_integer, media_rate_fraction):
        self.segment_duration = segment_duration
        self.media_time = media_time
        self.media_rate_integer = media_rate_integer
        self.media_rate_fraction = media_rate_fraction


class FtypBox(Box):
    """Обрабатывает 'ftyp' бокс, который определяет тип и совместимость файла."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.major_brand = ""
        self.minor_version = 0
        self.compatible_brands = []

    def parse(self):
        """Извлекает информацию о типе файла и его совместимости."""
        data = self.read_box_data()
        if len(data) < 8:
            raise ValueError("FTYP box data is too short.")
        self.major_brand = data[0:4].decode("utf-8")
        self.minor_version = struct.unpack(">I", data[4:8])[0]
        self.compatible_brands = [data[i:i+4].decode("utf-8") for i in range(8, len(data), 4)]

    def get_box_bytes(self):
        """Возвращает байты данных 'ftyp' бокса."""
        box_content = self.major_brand.encode('utf-8')
        box_content += struct.pack(">I", self.minor_version)
        for brand in self.compatible_brands:
            box_content += brand.encode('utf-8')
        return box_content

    def write_box_data(self):
        """Записывает 'ftyp' бокс в файл."""
        box_data = self.get_box_bytes()
        box_size = len(box_data) + BOX_HEADER_SIZE
        box_header = struct.pack(">I4s", box_size, self.name.encode('utf-8'))
        return box_header + box_data  # Возвращаем заголовок бокса и данные для записи


class HdlrBox:
    def __init__(self, box):
        self.box = box
        self.version = 0
        self.flags = 0
        self.handler_type = ""
        self.name = ""

    def parse(self):
        data = self.box.read_box_data()
        if len(data) < 24:  # Проверяем минимальную длину данных
            raise ValueError("Not enough data to parse HdlrBox")
        self.version = data[0]
        self.flags = int.from_bytes(data[1:4], "big")  # Исправлено для корректного чтения флагов
        self.handler_type = data[8:12].decode("utf-8")
        # Name начинается с 24-го байта до конца данных, учитывая 0-терминатор в конце строки
        self.name = data[24:-1].decode("utf-8").rstrip("\0")


class HmhdBox(Box):
    """Обрабатывает 'hmhd' бокс, специфичный для треков, содержащих подсказки."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.max_pdu_size = 0
        self.avg_pdu_size = 0
        self.max_bitrate = 0
        self.avg_bitrate = 0

    def parse(self):
        """Извлекает параметры подсказок из бокса."""
        data = self.read_box_data()
        self.version = data[0]
        self.max_pdu_size = struct.unpack(">H", data[1:3])[0]
        self.avg_pdu_size = struct.unpack(">H", data[3:5])[0]
        self.max_bitrate = struct.unpack(">I", data[5:9])[0]
        self.avg_bitrate = struct.unpack(">I", data[9:13])[0]


class MdatBox(Box):
    """Обрабатывает 'mdat' бокс, содержащий медиа-данные."""
    def __init__(self, data=None):
        super().__init__('mdat', len(data) + BOX_HEADER_SIZE, 0, None)
        self.data = data

    def write_box_data(self):
        """Записывает 'mdat' бокс в файл."""
        box_header = struct.pack(">I4s", self.size, b'mdat')
        return box_header + self.data


class MdhdBox:
    def __init__(self, box):
        self.box = box
        self.version = 0
        self.flags = 0
        self.creation_time = 0
        self.modification_time = 0
        self.timescale = 0
        self.duration = 0
        self.language = 0
        self.language_string = ""

    def parse(self):
        data = self.box.read_box_data()
        if len(data) < 24:  # Проверка длины для версии 0
            raise ValueError("Not enough data to parse MdhdBox")
        self.version = data[0]
        self.flags = int.from_bytes(data[1:4], "big")

        if self.version == 0:
            self.creation_time = struct.unpack(">I", data[4:8])[0]
            self.modification_time = struct.unpack(">I", data[8:12])[0]
            self.timescale = struct.unpack(">I", data[12:16])[0]
            self.duration = struct.unpack(">I", data[16:20])[0]
            self.language = struct.unpack(">H", data[20:22])[0]
            self.language_string = get_language_string(self.language)
        elif self.version == 1:
            self.creation_time = struct.unpack(">Q", data[4:12])[0]
            self.modification_time = struct.unpack(">Q", data[12:20])[0]
            self.timescale = struct.unpack(">I", data[20:24])[0]
            self.duration = struct.unpack(">Q", data[24:32])[0]
            self.language = struct.unpack(">H", data[32:34])[0]
            self.language_string = get_language_string(self.language)
        else:
            raise ValueError("Unsupported mdhd version")


class MdiaBox(Box):
    """Обрабатывает 'mdia' бокс, являющийся контейнером для медиа-информации трека."""
    def __init__(self, track_id, media_type):
        super().__init__('mdia')
        self.track_id = track_id
        self.media_type = media_type

    def parse(self):
        """Извлекает и обрабатывает вложенные боксы: 'mdhd', 'hdlr', 'minf'."""
        boxes = self.reader.read_boxes(self.start + BOX_HEADER_SIZE, self.size - BOX_HEADER_SIZE)
        for box in boxes:
            if box.name == "mdhd":
                self.mdhd = MdhdBox(box)
                self.mdhd.parse()
            elif box.name == "hdlr":
                self.hdlr = HdlrBox(box)
                self.hdlr.parse()
            elif box.name == "minf":
                self.minf = MinfBox(box)
                self.minf.parse()


class MinfBox(Box):
    """Обрабатывает 'minf' бокс, являющийся контейнером для более специфической медиа-информации."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.vmhd = None
        self.smhd = None
        self.hmhd = None
        self.stbl = None

    def parse(self):
        """Извлекает и обрабатывает вложенные боксы: 'vmhd', 'smhd', 'hmhd', 'stbl'."""
        boxes = self.reader.read_boxes(self.start + BOX_HEADER_SIZE, self.size - BOX_HEADER_SIZE)
        for box in boxes:
            if box.name == "vmhd":
                self.vmhd = VmhdBox(box)
                self.vmhd.parse()
            elif box.name == "smhd":
                self.smhd = SmhdBox(box)
                self.smhd.parse()
            elif box.name == "hmhd":
                self.hmhd = HmhdBox(box)
                self.hmhd.parse()
            elif box.name == "stbl":
                self.stbl = StblBox(box)
                self.stbl.parse()

class SmhdBox(Box):
    """Обрабатывает 'smhd' бокс, специфичный для аудио треков."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.balance = 0

    def parse(self):
        """Извлекает параметры аудио, включая баланс."""
        data = self.read_box_data()
        self.version = data[0]
        # Пропускаем флаги
        self.balance = struct.unpack(">h", data[4:6])[0] / 256  # Fixed point 8.8


class MoovBox(Box):
    """Обрабатывает 'moov' бокс, являющийся основным контейнером для метаданных файла."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.mvhd = None
        self.traks = []
        self.is_fragmented = False

    def parse(self):
        """Извлекает и обрабатывает вложенные боксы, включая 'mvhd' и все 'trak' боксы."""
        boxes = self.reader.read_boxes(self.start + BOX_HEADER_SIZE, self.size - BOX_HEADER_SIZE)
        for box in boxes:
            if box.name == "mvhd":
                self.mvhd = MvhdBox(box)
                self.mvhd.parse()
            elif box.name == "trak":
                trak = TrakBox(box)
                trak.parse()
                self.traks.append(trak)
            elif box.name == "mvex":
                self.is_fragmented = True

    def write_box_data(self):
        box_content = b''
        for box in self.boxes:
            box_content += box.write_box_data()
        box_size = len(box_content) + BOX_HEADER_SIZE
        self.size = box_size  # Обновляем размер с учетом содержимого
        box_header = struct.pack(">I4s", box_size, self.name.encode('utf-8'))
        return box_header + box_content


class MvhdBox(Box):
    def __init__(self, creation_time, modification_time, timescale, duration):
        super().__init__('mvhd')
        self.creation_time = creation_time
        self.modification_time = modification_time
        self.timescale = timescale
        self.duration = duration
        self.rate = 0x00010000  # Типичное значение
        self.volume = 0x0100  # Типичное значение

    def add_content(self):
        # Формирование содержимого mvhd бокса
        self.content = struct.pack(">IHHIIIIIIHHHHHHIIIIII",
                                   0,  # версия (0) и флаги
                                   self.creation_time, self.modification_time,
                                   self.timescale, self.duration,
                                   self.rate, self.volume,
                                   0, 0,  # зарезервировано
                                   0x00010000, 0, 0, 0, 0x00010000, 0,  # матрица
                                   0, 0, 0, 0, 0,  # pre_defined
                                   0xFFFFFFFF)  # next_track_ID

    def parse(self):
        """Извлекает общие метаданные из бокса."""
        data = self.read_box_data()
        self.version = data[0]
        index = 4  # Пропускаем флаги
        if self.version == 1:
            self.creation_time = struct.unpack(">Q", data[index:index+8])[0]
            index += 8
            self.modification_time = struct.unpack(">Q", data[index:index+8])[0]
            index += 8
            self.timescale = struct.unpack(">I", data[index:index+4])[0]
            index += 4
            self.duration = struct.unpack(">Q", data[index:index+8])[0]
        else:
            self.creation_time = struct.unpack(">I", data[index:index+4])[0]
            index += 4
            self.modification_time = struct.unpack(">I", data[index:index+4])[0]
            index += 4
            self.timescale = struct.unpack(">I", data[index:index+4])[0]
            index += 4
            self.duration = struct.unpack(">I", data[index:index+4])[0]
        index += 4  # Пропуск duration для версии 0
        self.rate = Fixed32(struct.unpack(">I", data[index:index+4])[0])
        index += 4
        self.volume = Fixed16(struct.unpack(">H", data[index:index+2])[0])



class StblBox(Box):
    """Обрабатывает 'stbl' бокс, являющийся контейнером для таблиц, описывающих трек."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.stsd = None
        self.stts = None
        self.stsc = None
        self.stsz = None
        self.stco = None

    def parse(self):
        """Извлекает и обрабатывает вложенные боксы, относящиеся к таблицам трека."""
        boxes = self.reader.read_boxes(self.start + BOX_HEADER_SIZE, self.size - BOX_HEADER_SIZE)
        for box in boxes:
            if box.name == "stsd":
                self.stsd = StsdBox(box)
                self.stsd.parse()
            elif box.name == "stts":
                self.stts = SttsBox(box)
                self.stts.parse()


class StsdBox(Box):
    """Обрабатывает 'stsd' бокс, содержащий декодеры специфичных медиа-форматов."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.entries = []

    def parse(self):
        """Извлекает информацию о декодерах из бокса."""
        data = self.read_box_data()
        self.version = data[0]
        entry_count = struct.unpack(">I", data[4:8])[0]
        offset = 8
        for _ in range(entry_count):
            # Здесь можно добавить логику для анализа каждой записи.
            # Важно отметить, что каждая запись может иметь различный формат в зависимости от типа медиа.
            pass

class CttsBox(Box):
    """Обрабатывает 'ctts' (composition time to sample) бокс, который содержит информацию о смещении времени композиции."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.entries = []

    def parse(self):
        """Извлекает таблицу смещения времени композиции для семплов."""
        data = self.read_box_data()
        self.version = data[0]
        entry_count = struct.unpack(">I", data[4:8])[0]
        offset = 8
        for _ in range(entry_count):
            sample_count, sample_offset = struct.unpack(">II", data[offset:offset+8])
            self.entries.append((sample_count, sample_offset))
            offset += 8

class StscBox(Box):
    """Обрабатывает 'stsc' (sample-to-chunk) бокс, описывающий распределение семплов по чанкам."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.entries = []

    def parse(self):
        """Извлекает правила, по которым семплы распределены по чанкам."""
        data = self.read_box_data()
        self.version = data[0]
        entry_count = struct.unpack(">I", data[4:8])[0]
        offset = 8
        for _ in range(entry_count):
            first_chunk, samples_per_chunk, sample_description_index = struct.unpack(">III", data[offset:offset+12])
            self.entries.append((first_chunk, samples_per_chunk, sample_description_index))
            offset += 12

class StszBox(Box):
    """Обрабатывает 'stsz' (sample size) бокс, содержащий информацию о размере каждого семпла."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.sample_size = 0
        self.sample_count = 0
        self.entries = []

    def parse(self):
        """Извлекает размеры семплов."""
        data = self.read_box_data()
        self.version = data[0]
        self.sample_size = struct.unpack(">I", data[4:8])[0]
        self.sample_count = struct.unpack(">I", data[8:12])[0]
        offset = 12
        if self.sample_size == 0:  # Если sample_size == 0, то размеры семплов различаются
            for _ in range(self.sample_count):
                sample_size = struct.unpack(">I", data[offset:offset+4])[0]
                self.entries.append(sample_size)
                offset += 4


class SttsBox(Box):
    """Обрабатывает 'stts' (sample table time-to-sample) бокс, описывающий время семплов."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.entries = []

    def parse(self):
        """Извлекает таблицу времени к семплам."""
        data = self.read_box_data()
        self.version = data[0]
        entry_count = struct.unpack(">I", data[4:8])[0]
        offset = 8
        for _ in range(entry_count):
            sample_count, sample_delta = struct.unpack(">II", data[offset:offset+8])
            self.entries.append((sample_count, sample_delta))
            offset += 8


class StcoBox(Box):
    """Обрабатывает 'stco' (chunk offset) бокс, указывающий на местоположение каждого чанка в файле."""
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.version = 0
        self.entries = []

    def parse(self):
        """Извлекает смещения чанков в файле."""
        data = self.read_box_data()
        self.version = data[0]
        entry_count = struct.unpack(">I", data[4:8])[0]
        offset = 8
        for _ in range(entry_count):
            chunk_offset = struct.unpack(">I", data[offset:offset+4])[0]
            self.entries.append(chunk_offset)
            offset += 4


class TkhdBox(Box):
    def __init__(self, track_id, duration, width, height, creation_time=0, modification_time=0, layer=0, alternate_group=0, volume=0x0100):
        super().__init__('tkhd')
        self.version = 0  # предполагаем, что используем версию 0 для упрощения
        self.flags = 3  # значение по умолчанию для активного трека
        self.creation_time = creation_time
        self.modification_time = modification_time
        self.track_id = track_id
        self.duration = duration
        self.layer = layer
        self.alternate_group = alternate_group
        self.volume = volume
        self.matrix = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000]  # стандартная матрица
        self.width = width
        self.height = height

    def parse(self):
        data = self.box.read_box_data()
        self.version = data[0]
        self.flags = int.from_bytes(data[1:4], "big")
        index = 4
        if self.version == 0:
            self.creation_time = struct.unpack(">I", data[index:index + 4])[0]
            index += 4
            self.modification_time = struct.unpack(">I", data[index:index + 4])[0]
            index += 4
            self.track_id = struct.unpack(">I", data[index:index + 4])[0]
            index += 8  # Пропускаем 4 байта зарезервированного пространства
            self.duration = struct.unpack(">I", data[index:index + 4])[0]
            index += 4
        elif self.version == 1:
            self.creation_time = struct.unpack(">Q", data[index:index + 8])[0]
            index += 8
            self.modification_time = struct.unpack(">Q", data[index:index + 8])[0]
            index += 8
            self.track_id = struct.unpack(">I", data[index:index + 4])[0]
            index += 8  # Пропускаем 4 байта зарезервированного пространства
            self.duration = struct.unpack(">Q", data[index:index + 8])[0]
            index += 8
        else:
            raise ValueError("Unsupported tkhd version")

        # Пропускаем зарезервированные байты
        index += 8
        self.layer = struct.unpack(">H", data[index:index + 2])[0]
        index += 2
        self.alternate_group = struct.unpack(">H", data[index:index + 2])[0]
        index += 2
        self.volume = struct.unpack(">H", data[index:index + 2])[0] / 256
        index += 2
        # Пропускаем зарезервированные байты
        index += 2
        self.matrix = struct.unpack(">9I", data[index:index + 36])  # Матрица трансформации
        index += 36
        self.width = struct.unpack(">I", data[index:index + 4])[0] / 65536
        index += 4
        self.height = struct.unpack(">I", data[index:index + 4])[0] / 65536

    def write_box_data(self, output):
        header = struct.pack(">I4s", 0, self.name.encode('utf-8'))  # размер будет обновлен
        data = struct.pack(">BBHIIIHHI9IHHII",
                           self.version, 0, 0,  # версия и флаги (2 байта флаги здесь нули)
                           self.creation_time, self.modification_time,
                           self.track_id, 0,  # зарезервировано
                           self.duration,
                           0, 0,  # зарезервировано
                           self.layer, self.alternate_group,
                           self.volume, 0,  # зарезервировано
                           *self.matrix,
                           int(self.width * 65536), int(self.height * 65536))  # размеры в fixed-point
        self.size = len(header) + len(data) - 8  # корректировка размера, вычитаем размер заголовка
        output.write(
            struct.pack(">I", self.size) + header[4:] + data)  # обновляем заголовок с новым размером и записываем

class TrakBox(Box):
    """Обрабатывает 'trak' бокс, являющийся контейнером для информации о треке."""
    def __init__(self, track_id, duration, width, height, media_type):
        super().__init__('trak')
        self.track_id = track_id
        self.duration = duration
        self.width = width
        self.height = height
        self.media_type = media_type  # 'video' или 'audio'
        # Создание TkhdBox с предоставленными параметрами
        self.tkhd = TkhdBox(track_id=self.track_id, duration=self.duration, width=self.width, height=self.height)
        # Предполагается, что MdiaBox принимает тип медиа и другие параметры, необходимые для генерации 'mdia' бокса
        self.mdia = MdiaBox(track_id=self.track_id, media_type=self.media_type)

    def parse(self):
        """Извлекает и обрабатывает вложенные боксы 'tkhd' и 'mdia'."""
        boxes = self.reader.read_boxes(self.start + BOX_HEADER_SIZE, self.size - BOX_HEADER_SIZE)
        for box in boxes:
            if box.name == "tkhd":
                self.tkhd = TkhdBox(box)
                self.tkhd.parse()
            elif box.name == "mdia":
                self.mdia = MdiaBox(box)
                self.mdia.parse()

    def write_box_data(self, output):
        start_position = output.tell()
        output.write(struct.pack(">I4s", 0, b'trak'))  # Плейсхолдер для размера и имя бокса

        # Записываем содержимое TkhdBox
        self.tkhd.write_box_data(output)

        # Записываем содержимое MdiaBox
        self.mdia.write_box_data(output)

        # Обновление размера 'trak' бокса
        end_position = output.tell()
        box_size = end_position - start_position
        output.seek(start_position)
        output.write(struct.pack(">I", box_size))  # Обновление размера бокса
        output.seek(end_position)  # Возвращение к концу бокса для продолжения записи

class UuidBox(Box):
    """
    Обрабатывает 'uuid' бокс, который может содержать пользовательские данные.
    UUID боксы используются для расширения стандарта без вмешательства в официальную спецификацию MP4.
    """
    def __init__(self, box):
        super().__init__(box.name, box.size, box.start, box.reader)
        self.uuid = []
        self.data = []

    def parse(self):
        """Извлекает UUID и данные, связанные с ним."""
        data = self.read_box_data()
        if len(data) < UUID_SIZE:
            raise ValueError("Data in UUID box is too short to contain valid UUID")
        self.uuid = data[:UUID_SIZE]
        self.data = data[UUID_SIZE:]


class VmhdBox:
    def __init__(self, box):
        self.Box = box
        self.version = 0
        self.Flags = 0
        self.GraphicsMode = 0
        self.OpColor = 0

    def parse(self):
        data = self.Box.read_box_data()
        self.version = data[0]
        self.Flags = struct.unpack(">I", data[0:4])[0]
        self.GraphicsMode = struct.unpack(">H", data[4:6])[0]
        self.OpColor = struct.unpack(">H", data[6:8])[0]


class Fixed16:
    def __init__(self, value):
        self.value = value

    def to_float(self):
        # Разделяем на целую и дробную часть.
        # Сдвигаем биты на 8 вправо для получения целой части
        # и применяем маску 0xFF для получения дробной части
        integer_part = self.value >> 8
        fractional_part = self.value & 0xFF
        # Преобразуем дробную часть в десятичную дробь
        fractional_decimal = fractional_part / 256
        return integer_part + fractional_decimal

    def __str__(self):
        return str(self.to_float())


class Fixed32:
    def __init__(self, value):
        self.value = value

    def to_float(self):
        # Разделяем на целую и дробную часть.
        # Сдвигаем биты на 16 вправо для получения целой части
        # и применяем маску 0xFFFF для получения дробной части
        integer_part = self.value >> 16
        fractional_part = self.value & 0xFFFF
        # Преобразуем дробную часть в десятичную дробь
        fractional_decimal = fractional_part / 65536
        return integer_part + fractional_decimal

    def __str__(self):
        return str(self.to_float())


def get_duration_string(duration, timescale):
    """
    Преобразует продолжительность из единиц времени MP4 в читаемый формат чч:мм:сс:мс.
    """
    duration_sec = duration / timescale

    hours = duration_sec // 3600
    duration_sec %= 3600

    minutes = duration_sec // 60
    seconds = duration_sec % 60

    return "{:02.0f}:{:02.0f}:{:02.0f}".format(hours, minutes, seconds)


def write_mp4_file(encoded_images, file_path):
    mp4_writer = Mp4Writer(file_path)

    ftyp_data = FtypBox(Box('ftyp', 24, 0, None))
    ftyp_data.major_brand = 'isom'
    ftyp_data.minor_version = 512
    ftyp_data.compatible_brands = ['isom', 'iso2', 'avc1', 'mp41']
    mp4_writer.add_box(ftyp_data)

    # Создание и добавление mdat бокса с медиаданными
    mdat_box = MdatBox(encoded_images)
    mp4_writer.add_box(mdat_box)

    # Запись файла
    mp4_writer.write()
