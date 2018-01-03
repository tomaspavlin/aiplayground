import patterns

def _compute_difference(o1, o2):

    if True:
        o1 = [0 if a < 0.5 else 1 for a in o1]
        o2 = [0 if a < 0.5 else 1 for a in o2]

    return sum([pow(a - b, 2) for a, b in zip(o1, o2)]) / 100.0

def get_diff(o1, o2):
    return _compute_difference(o1, o2)


def test_network(net):
    ret = 0
    pats = patterns.get_all_patterns()

    count = 0

    for pat in pats:
        i = pat

        o = net.propagate(i)
        diff = _compute_difference(i, o)
        ret += diff

        count += 1



    ret /= count
    return ret
