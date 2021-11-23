

def tail(path):
    return path.split('/')[-1]

def parse_flag_from_string(string, flag, end="_"):
    return string.split(flag)[1].split(end)[0]