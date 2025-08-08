"""_summary_"""

def bytes_to_tuple(bytes: bytes):
    return tuple(map(int.to_bytes, bytes))
