import concurrent.futures

from OracleSolver import OracleSolver
from SystemData import SystemData
from TestSuite import *
from AlternativePaths import AlternativePaths, computeAlts
from CTTmetrics import getSPLOTsuites, getNumberOfSPLOTModels, smoothLinearApprox, computeCorrelation
import matplotlib.pyplot as plt


def printOriginalVsAlterativePaths(originalTestSuite, alternatives):
    original = originalTestSuite.getUnorderedTestSuite()
    prevTest = {f: -1 * abs(original[0][f]) for f in original[0] if f != "Context" and f != "Feature"}
    width = max([len(p) for p in alternatives])
    print("1 & " + originalTestSuite.getPrintableLine(prevTest, original[0]) + " & \\multicolumn{"+str(width)+"}{l|}{\\emph{No transition coverage in the first test configuration.}} \\\\\\hline  ")
    for i in range(1, len(original)):
        paths = alternatives[i-1]
        lengthAndCost = [p.getShortenedLengthAndCost() for p in paths]
        if len(lengthAndCost) == 0:
            print(str(i+1) + " & " + originalTestSuite.getPrintableLine(original[i-1], original[i]) + " & \\multicolumn{"+str(width)+"}{l|}{\\emph{No alternative execution path is necessary.}} \\\\\\hline  ")
            continue
        maxLine = max([t[0] for t in lengthAndCost])-1
        toPrint = [" & " for i in range(maxLine)]
        toPrint[0] = str(i+1) + " & \\multirow{" + str(maxLine) + "}{=}{" + originalTestSuite.getPrintableLine(original[i-1], original[i]) + "} "
        for k in range(len(paths)):
            #print(k)
            pathSuite = paths[k]
            path = pathSuite.getShortenedTestSuite()
            for j in range(maxLine):
                toPrint[j] += "& "
                if j < len(path)-1:
                    if k == len(paths)-1 and len(paths) < width:
                        toPrint[j] += "\\multicolumn{" + str(width - len(paths) + 1) + "}{l|}{" + pathSuite.getPrintableLine(path[j], path[j+1]) + "}"
                    else:
                        # print("len path", len(path), " len paths", len(paths), " current j ", j, " len toprint", len(toPrint))
                        #print(path[j+1])
                        toPrint[j] += pathSuite.getPrintableLine(path[j], path[j+1])

        toPrint = [tp + " \\\\\\cline{3-" + str(width+2) + "}" for tp in toPrint[:-1]] + [(toPrint[-1] + " \\\\\\hline")]
        for t in toPrint:
            print(t)


def RISpath():
    models = "../data/RIS-FOP/"
    s = SystemData(featuresFile=models + 'features.txt')
    storage = models + "TestSuitesCTT/"
    altsStorage = models + "AlternativePaths/alts"
    iteration = 1
    testsuite = computeCTTSuite(storage, iteration, s, recompute=False, verbose=True)
    testsuite.printLatexTransitionForm()
    paths, undetectables = computeAlts(altsStorage, s, testsuite.getUnorderedTestSuite(), iteration, states=4, recompute=False)
    printOriginalVsAlterativePaths(testsuite, paths)

    lengthAndCost = [t.getShortenedLengthAndCost() for p in paths for t in p]
    averageNumberOfPaths = sum([len(p) for p in paths]) / len(paths)
    totalStates = sum([t[0] - 2 for t in lengthAndCost])
    averageLength = totalStates / len(lengthAndCost)
    totalCost = sum([t[1] for t in lengthAndCost])
    averageCost = totalCost / len(lengthAndCost)
    print("number of groups of paths is", len(paths), ", average number of paths is", averageNumberOfPaths,
          "their length is on average ", averageLength, "their cost is on average", averageCost)
    print("total number of states is", totalStates, "total cost is", totalCost)

def debugging(filename, iteration=0):
    models = "../data/SPLOT/SPLOT-NEW/"
    storageAlts = "../data/SPLOT/SPLOT-NEW/SPLOT-Alts/"
    s = SystemData(featuresFile=models+"SPLOT-txt/"+filename, extraConstraints=models+"SPLOT-txtconstraints/"+filename)
    storage = models + "SPLOT-TestSuitesCTT/" + filename[:-4] + "-1&2&3-"
    testsuite = computeCTTSuite(storage, iteration, s, recompute=False, verbose=True)
    #testsuite.printLatexTransitionForm()
    #print(testsuite.interactionTransitionCoverageEvolution())
    suite = testsuite.getUnorderedTestSuite()
    print("number of nodes is", len(s.getNodes()), ", length of suite is", len(suite))
    states = 4
    paths, undetectables = computeAlts(storageAlts + filename[:-4], s, suite, tag=iteration)

    #print("paths :", paths)
    #print("len coverage :", len(coveredTransitions))
    lengthAndCost = [t.getShortenedLengthAndCost() for p in paths for t in p]
    averageNumberOfPaths = sum([len(p) for p in paths])/len(paths)
    totalStates = sum([t[0]-2 for t in lengthAndCost])
    averageLength = totalStates/len(lengthAndCost)
    totalCost = sum([t[1] for t in lengthAndCost])
    averageCost = totalCost/len(lengthAndCost)
    print("number of groups of paths is", len(paths), ", average number of paths is", averageNumberOfPaths, "their length is on average ", averageLength, "their cost is on average", averageCost)
    print("total number of states is", totalStates, "total cost is", totalCost)
    print("undetectables : ", str(undetectables*100) + "%")
    #print("len undecomposable :", len(alts.getNondecomposableTransitions()), "vs len decomposable :", len(alts.getDecomposableTransitions()))

    #oracle = OracleSolver(s, states)
    #testSuite = oracle.createPath(suite[0], suite[1])
    #if testSuite is not None:
    #    testSuite = testSuite.getUnorderedTestSuite()
    #    for i in range(len(testSuite)):
    #        print(testSuite[i])

def oracleSPLOTmetrics(rangeCategory, states=None, recompute=False, verbose=False):
    if states is None:
        states = [2, 3, 4, 5]
    steps = [0 for s in states]
    cost = [0 for s in states]
    NAltsPaths = [0 for s in states]
    NAltsPerGroup = [0 for s in states]
    undetectables = [0 for s in states]
    max_iterations = 3
    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCTT/"
    storageAlts = "../data/SPLOT/SPLOT-NEW/SPLOT-Alts/"
    quty = 0
    total = getNumberOfSPLOTModels(rangeCategory)
    print("Computing model " + "0" + "/" + str(total) + " (category: " + str(rangeCategory) + ")", flush=True, end='')
    init = 0
    stop = 35
    sizesOracle = {}
    sizesCTT = {}
    for filename in os.listdir(modelFiles):
        #init += 1
        #if init == stop:
        #    break
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        if (rangeCategory[0] <= len(s.getFeatures()) < rangeCategory[1] and transitionExist(s)):
            quty += 1
            tempStorageCTT = storageCTT + filename[:-4] + "-1&2&3-"
            for iteration in range(max_iterations):
                print("\rComputing model " + str(quty) + "/" + str(total), iteration+1, "/", max_iterations, " (category: " + str(rangeCategory) + "), model " + str(filename), flush=True, end='')
                computedSuite = computeCTTSuite(tempStorageCTT, iteration, s, recompute=False)
                size = len(s.getFeatures())
                if size in sizesCTT:
                    sizesCTT[size].append(computedSuite.getLength())
                else:
                    sizesCTT[size] = [computedSuite.getLength()]

                currSuite = computedSuite.getUnorderedTestSuite()
                for state in states:
                    id = state-min(states)
                    tempstorageAlts = storageAlts + filename[:-4]
                    paths, undetectable = computeAlts(tempstorageAlts, s, currSuite, tag=iteration, states=state, recompute=recompute)
                    undetectables[id] += undetectable
                    lengthAndCost = [t.getShortenedLengthAndCost() for p in paths for t in p]
                    #print("Max steps :", max([t[0] - 2 for t in lengthAndCost]), " sum of steps : ", sum([t[0] - 2 for t in lengthAndCost]))
                    #averageNumberOfPaths = sum([len(p) for p in paths]) / len(paths)
                    steps[id] += sum([t[0] - 2 for t in lengthAndCost])
                    cost[id] += sum([t[1] for t in lengthAndCost])
                    NAltsPaths[id] += sum([1 for p in paths for t in p])
                    NAltsPerGroup[id] += sum([len(p) for p in paths])/len(paths)
                    #toAddToSizesOracle = sum([t[0] - 2 for t in lengthAndCost])
                    toAddToSizesOracle = sum([1 for p in paths for t in p])
                    if size in sizesOracle:
                        sizesOracle[size].append(toAddToSizesOracle)
                    else:
                        sizesOracle[size] = [toAddToSizesOracle]

    computeCorrelation(sizesCTT, sizesOracle)
    normalise = quty*max_iterations
    steps = [round(s/normalise, 1) for s in steps]
    cost = [round(c/normalise, 1) for c in cost]
    undetectables = [str(round(u*100 / normalise, 1)) + "%" for u in undetectables]

    NAltsPaths = [round(na/normalise, 1) for na in NAltsPaths]
    NAltsPerGroup = [round(napg/normalise, 2) for napg in NAltsPerGroup]
    delimiter = " & "
    toPrint = ""
    ranges = str(rangeCategory[0]) + "-" + str(rangeCategory[1] - 1)
    if len(states) > 1:
        font = {'size': 16}
        plt.rc('font', **font)
        plt.xlabel("N")
        plt.ylabel("Number of intermediate configurations")
        x1_smooth, y1_smooth = smoothLinearApprox(states, steps)
        plt.plot(x1_smooth, y1_smooth, '-')
        plt.show()
    else:
        for arg in [ranges, total] + steps + NAltsPaths + NAltsPerGroup+ cost + undetectables:
            toPrint += str(arg) + delimiter
        toPrint = toPrint.replace("%", "\%")
        print("\r" + toPrint[:-2] + " \\\\\\hline", flush=True)


if __name__ == '__main__':
    categories = [[10, 20], [20, 30], [30, 40], [40, 50], [50, 70], [70, 100]]
    #debugging("model_20130510_203163945.txt", 2)
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    futures = []
    #    futures.append(executor.submit(oracleSPLOTmetrics, [10, 30], [2, 3, 4, 5, 6], recompute=False, verbose=True))
    #    futures.append(executor.submit(oracleSPLOTmetrics, [30, 40], [2, 3, 4, 5, 6], recompute=False, verbose=True))
    #    futures.append(executor.submit(oracleSPLOTmetrics, [50, 60], [2, 3, 4, 5, 6], recompute=False, verbose=True))
    #    futures.append(executor.submit(oracleSPLOTmetrics, [70, 80], [2, 3, 4, 5, 6], recompute=False, verbose=True))
    #    futures.append(executor.submit(oracleSPLOTmetrics, [80, 90], [2, 3, 4, 5, 6], recompute=False, verbose=True))
    #    futures.append(executor.submit(oracleSPLOTmetrics, [90, 100], [2, 3, 4, 5, 6], recompute=False, verbose=True))
    #    done = 0
    #    for f in futures:
    #        done += 1
    #        f.result()
    #       print("done :", done)

    oracleSPLOTmetrics([10, 100], [2, 3, 4, 5, 6], recompute=False, verbose=True)

    for r in categories:
        oracleSPLOTmetrics(r, [6], recompute=False, verbose=True)
