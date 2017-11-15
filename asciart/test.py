import patterns

cycles = 100


def _compute_difference(o1, o2):
    return sum([pow(a - b, 2) for a, b in zip(o1, o2)])


def test_network(net):
    ret = 0
    for i in range(cycles):
        i, o = patterns.get_pattern()

        o2 = net.propagate(i)
        diff = _compute_difference(o, o2)
        ret += diff

    ret /= cycles
    return ret
