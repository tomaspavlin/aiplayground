import patterns

cycles = 200


def _compute_difference(o1, o2):
    return sum([pow(a - b, 2) for a, b in zip(o1, o2)])


def test_network_for_correct(net):
    ret = 0
    for i in range(cycles):
        i, o = patterns.getRandCorrectPattern()

        o2 = net.propagate(i)
        diff = _compute_difference(o, o2)
        ret += diff

    ret /= cycles
    return ret


def test_network_for_incorrect(net):
    ret = 0
    for i in range(cycles):
        i, o = patterns.getRandIncorrectPattern()

        o2 = net.propagate(i)
        diff = _compute_difference(o, o2)
        ret += diff

    ret /= cycles
    return ret
