# common definitions

# nucleotide code Base, ASCII
NCB_ASC = {
    ('A', 'A'): 'A',                       # Adenine
    ('C', 'C'): 'C',                       # Cytosine
    ('G', 'G'): 'G',                       # Guanine
    ('T', 'T'): 'T',                       # Thymine

    ('A', 'G'): 'R',
    ('C', 'T'): 'Y',
    ('G', 'C'): 'S',
    ('A', 'T'): 'W',
    ('G', 'T'): 'K',
    ('A', 'C'): 'M',

    ('G', 'A'): 'R',
    ('T', 'C'): 'Y',
    ('C', 'G'): 'S',
    ('T', 'A'): 'W',
    ('T', 'G'): 'K',
    ('C', 'A'): 'M',

    ('-', 'A'): 'B',
    ('-', 'C'): 'D',
    ('-', 'G'): 'H',
    ('-', 'T'): 'V',

    ('A', '-'): 'B',
    ('C', '-'): 'D',
    ('G', '-'): 'H',
    ('T', '-'): 'V',

    ('-', '-'): '-',
    ('N', 'N'): 'N'}

# nucleotide code base, ascii number
NCB_INT = {
    ('N', 'N'): 0x00,           # Anything
    ('A', 'A'): 0x01,           # Adenine
    ('C', 'C'): 0x02,           # Cytosine
    ('G', 'G'): 0x03,           # Guanine
    ('T', 'T'): 0x04,           # Thymine

    ('A', 'G'): 0x05,
    ('C', 'T'): 0x06,
    ('G', 'C'): 0x07,
    ('A', 'T'): 0x08,
    ('G', 'T'): 0x09,
    ('A', 'C'): 0x0A,

    ('G', 'A'): 0x05,
    ('T', 'C'): 0x06,
    ('C', 'G'): 0x07,
    ('T', 'A'): 0x08,
    ('T', 'G'): 0x09,
    ('C', 'A'): 0x0A,

    ('A', '-'): 0x0B,
    ('C', '-'): 0x0C,
    ('G', '-'): 0x0D,
    ('T', '-'): 0x0E,

    ('-', 'A'): 0x0B,
    ('-', 'C'): 0x0C,
    ('-', 'G'): 0x0D,
    ('-', 'T'): 0x0E,

    ('-', '-'): 0x0F}

NCB = {None: NCB_ASC, str: NCB_ASC, int: NCB_INT}

# chromosome keys
CKY = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', 'X', 'Y']
