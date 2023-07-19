"""Some convenient functions for processing data files"""

from typing import Any, Optional


def tail(path: str) -> str:
    """Returns tail of a UNIX path."""
    return path.split('/')[-1]


def parse_token(string: str,
                token: str,
                sep: str = "_",
                val_sep: str = "-",
                dtype: Optional[Any] = None) -> Any:
    """Returns the value associated with a token in a UNIX path."""
    token_padded = sep + token + val_sep
    out = string.split(token_padded)[1].split(sep)[0]
    if dtype is not None:
        return dtype(out)
    else:
        return out
