import patterns

def _compute_difference(o1, o2):
    return sum([pow(a - b, 2) for a, b in zip(o1, o2)]) / 100


def test_network(net):
    ret = 0
    pats = patterns.get_all_patterns()

    count = 0

    for pat in pats:
        i = o = pat

        o2 = net.propagate(i)
        diff = _compute_difference(o, o2)
        ret += diff

        count += 1



    ret /= count
    return ret
