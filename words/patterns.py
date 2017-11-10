import numpy as np

words = [
    "begin",
    "end",
    "and",
    "else",
    "for",
    "ananas",
    "table"
]

# probability of wrong input bit change. 0 = no change, 1 = change completely, must be > 0
changeProb = 0.1

def get_words():
    return words

def interpret_output_i(output):
    max = 0
    max_i = 0
    for i in range(len(output)):
        if output[i] >= max:
            max = output[i]
            max_i = i

    return max_i


def encodeChar(c):
    pos = " abcdefghijklmnopqrstuvwxyz".index(c)

    ret = []
    for i in range(5):
        cur = pos & 1
        pos = pos >> 1
        ret = [cur] + ret

    return ret

def encodeWord(w):
    ret = []
    for i in range(6):
        if len(w) > i:
            ret = ret + encodeChar(w[i])
        else:
            ret = ret + encodeChar(' ')

    return ret

def createOutput(index):
    ret = np.zeros(8)
    ret[index] = 1
    return ret

def getRandCorrectPattern():
    i = np.random.randint(len(words))
    input = encodeWord(words[i])
    output = createOutput(i)

    return input, output



def getRandIncorrectPattern():
    def isInputCorrect(input):
        return any([input == encodeWord(w) for w in words])

    input, _ = getRandCorrectPattern()

    # modify input
    while isInputCorrect(input):
        for i in range(len(input)):
            if np.random.random() < changeProb:
                input[i] = np.random.randint(2)

    # get output
    output = createOutput(7)

    return input, output

if __name__ == "__main__":
    print "pes:", encodeWord("pes")
    print createOutput(7)
    print "correct:", getRandCorrectPattern()
    print "incorrect:", getRandIncorrectPattern()

