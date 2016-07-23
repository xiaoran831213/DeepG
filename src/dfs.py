# common definitions

# nucleotide code Base, ASCII
NCB = {
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

# nucleotide code base, Integer
NCB_ASC2INT = {
    'N': 0x00,                            # Anything
    'A': 0x01,                            # Adenine
    'C': 0x02,                            # Cytosine
    'G': 0x03,                            # Guanine
    'T': 0x04,                            # Thymine

    'R': 0x05,
    'Y': 0x06,
    'S': 0x07,
    'W': 0x08,
    'K': 0x09,
    'M': 0x0A,

    'B': 0x0B,
    'D': 0x0C,
    'H': 0x0D,
    'V': 0x0E,

    '-': 0x0F}

# chromosome keys
CKY = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', 'X', 'Y']
