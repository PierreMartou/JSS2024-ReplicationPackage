import random



def cover(configs, coverage):
    for t in coverage.copy():
        for i in range(len(configs[:-1])):
            now = configs[i]
            next = configs[i+1]
            if -t[0] == now[0] and -t[1] == now[1] and t[0] == next[0] and t[1] == next[1]:
                coverage.remove(t)

    return coverage

def averageForCoverage():
    configs = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    coverage = configs.copy()
    random.shuffle(configs)
    #configs = [(-1, -1), (1, 1), (-1, -1), (-1, 1), (1, -1), (-1, 1)]

    quty = 0
    while len(coverage) != 0:
        quty += 1
        coverage = cover(configs, coverage)
        random.shuffle(configs)
    return quty

averageQuty = 0
iterations = 100000
for i in range(iterations):
    averageQuty += averageForCoverage()
print(averageQuty/iterations)

