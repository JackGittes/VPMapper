def check_tuple(_in)->(int, int):
    assert isinstance(_in, int) or isinstance(_in, tuple), \
        "Input must be a tuple or an integer."
    if isinstance(_in, tuple):
        res = _in
    else:
        res = (_in, _in)
    return res


