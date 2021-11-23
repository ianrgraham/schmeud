

def tail(path):
    return path.split('/')[-1]

def parse_flag_from_string(string, flag):
    return string.split(f'_{flag}')[1].split("_")[0]