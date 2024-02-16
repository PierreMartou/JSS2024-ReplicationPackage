import scipy.stats

from TestSuite import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as scistats
import concurrent.futures


def computeEvolutionAverage(data, iterations):
    evolutionAverage = []
    for evolution in data:
        for i in range(len(evolution)):
            number = evolution[i]
            if i == len(evolutionAverage):
                evolutionAverage.append([100.0 - number])
            else:
                evolutionAverage[i].append(100.0 - number)
    evolutionAverage = [100.0 - sum(evolution) / len(evolution) for evolution in evolutionAverage]
    return evolutionAverage


def computeSwitchesAverage(data, iterations):
    switchesAverage = []
    for switches in data:
        for i in range(len(switches)):
            number = switches[i]
            if i == len(switchesAverage):
                switchesAverage.append([number])
            else:
                switchesAverage[i].append(number)
    evolutionAverage = [float(sum(switches)) / iterations for switches in switchesAverage]
    return evolutionAverage


def smoothLinearApprox(x, y):
    x_smooth = np.linspace(min(x), max(x), 500)
    f = interpolate.interp1d(x, y, kind="linear")
    y_smooth = f(x_smooth)
    return x_smooth, y_smooth


def RISEvaluation(iterations, mode=None, recompute=False, latex=False, plot=None):
    models = "../data/RIS-FOP/"
    s = SystemData(featuresFile=models + 'features.txt')
    font = {'size': 16}
    plt.rc('font', **font)
    # plt.set_tight_layout(True)
    #label = ["Baseline", "Improved"]
    label = ["Baseline algorithm"]
    lines = ["-"] #, "--"]
    lines2 = ["-."] #, ":"]
    linesCoverage = [lines, lines2]
    costAverages = []
    costVariances = []
    sizeAverages = []
    sizeVariances = []

    if mode is None:
        print("PLEASE INDICATE WHICH MODES IN RISEvaluation.")
        return
    modes = mode

    for mode in modes:
        allSize = []
        allCosts = []
        allInteractionEvolution = []
        allTransitionEvolution = []
        allSwitchesEvolution = []

        i_filter, w_la, w_c = recognizeMode(mode)
        storage = models + "TestSuitesCTT/testSuite" + str(mode) + "-"

        for i in range(iterations):
            testSuite = computeCTTSuite(storage, i, s, interaction_filter=i_filter, weight_lookahead=w_la,
                                        weight_comparative=w_c, recompute=recompute)
            allSize.append(testSuite.getLength())
            allCosts.append(testSuite.getCost())
            if plot is not None:
                interactionEvolution, transitionEvolution, switchesEvolution = testSuite.interactionTransitionCoverageEvolution()
                allSwitchesEvolution.append(switchesEvolution)
                allInteractionEvolution.append(interactionEvolution)
                allTransitionEvolution.append(transitionEvolution)

        costAverages.append(round(sum(allCosts) / len(allCosts), 1))
        costVariances.append(round(float(np.std(allCosts)), 1))

        sizeAverages.append(round(sum(allSize) / len(allSize), 1))
        sizeVariances.append(round(float(np.std(allSize)), 1))

    delimiter = ""
    ender = ""
    if latex:
        delimiter = " & "
        ender = " \\\\\\hline"
    else:
        print("Improvements \t Size \t Cost")
        delimiter = "\t"

    for i in range(len(modes)):
        if plot is None:
            print("Improvement " + str(modes[i]) + delimiter + str(sizeAverages[i]) + " (+-" + str(sizeVariances[i]) + ")" + delimiter + str(costAverages[i]) + " (+-" + str(costVariances[i]) + ")" + ender)

        if plot is not None:
            print("Mode " + str(modes[i]) + " - Cost average (std deviation): " + str(costAverages[i]) + " (+-" + str(
                costVariances[i]) + ")")
            print("Mode " + str(modes[i]) + " - Size average (std deviation): " + str(sizeAverages[i]) + " (+-" + str(
                sizeVariances[i]) + ")")

            interactionEvolutionAverage = computeEvolutionAverage(allInteractionEvolution, iterations)
            transitionEvolutionAverage = computeEvolutionAverage(allTransitionEvolution, iterations)
            switchesEvolutionAverage = computeSwitchesAverage(allSwitchesEvolution, iterations)

            cutoffy1 = sum([1 for i in interactionEvolutionAverage if i < 100.0])
            y1 = [0] + interactionEvolutionAverage[:cutoffy1] + [100]

            cutoffy2 = sum([1 for i in transitionEvolutionAverage if i < 100.0])
            y2 = transitionEvolutionAverage[:cutoffy2] + [100]

            switches_cutoff = sum([1 for i in switchesEvolutionAverage if i > 0.05])
            y3 = switchesEvolutionAverage[:switches_cutoff]

            x1 = np.linspace(0, len(y1) + 1, num=len(y1))
            x2 = np.linspace(1, len(y2) + 1, num=len(y2))
            x3 = np.linspace(1, len(y3) + 1, num=len(y3))
            x1_smooth, y1_smooth = smoothLinearApprox(x1, y1)
            x2_smooth, y2_smooth = smoothLinearApprox(x2, y2)
            x3_smooth, y3_smooth = smoothLinearApprox(x3, y3)
            # param1 = np.polyfit(x1, y1, 1)
            # y1_smooth = np.polyval(param1, x_new1)
            # popt = curve_fit(f_log, x1, y1)
            # a1 = popt[0]
            # y1_smooth = [f_log(x, *a1) for x in x_new1]
            # param2 = np.polyfit(x2, y2, 1)
            # y2_smooth = np.polyval(param2, x_new2)
            if plot == 0:
                # font = {'family': 'normal', 'size': 16}
                plt.plot(x1_smooth, y1_smooth, linesCoverage[0][modes.index(mode)],
                         label=label[modes.index(mode)] + " Interaction coverage ")
                plt.plot(x2_smooth, y2_smooth, linesCoverage[1][modes.index(mode)],
                         label=label[modes.index(mode)] + " Transition coverage ")
                plt.legend()
                # plt.legend(['interaction coverage', 'transition coverage evolution'])
                # plt.title('Evolution of the interaction/transition coverage of a test suite', fontsize=14)
                plt.xlabel('Test number')
                plt.ylabel('Coverage (%)')
                # test = [1, 3, 5, 7, 9, 11, 13, 15]
                # plt.xticks(test, test)
            elif plot == 1:
                # font = {'family': 'normal', 'size': 16}
                plt.plot(x3_smooth, y3_smooth, lines[modes.index(mode)], label=label[modes.index(mode)])
                plt.legend()
                # plt.legend(['interaction coverage', 'transition coverage evolution'])
                # plt.title('Number of switches in a test suite', fontsize=14)
                plt.xlabel('Test number')
                plt.ylabel('Number of switches')
                # test = [1, 3, 5, 7, 9, 11, 13, 15]
                # plt.xticks(test, test)
    if not latex:
        print("We drew our conclusions from the decrease in average size, cost and standard deviation from No Improvement to Improvement 1 & 2 & 3.")
        sizereduc = (sizeAverages[0]-sizeAverages[-1]) / sizeAverages[0]
        costreduc = (costAverages[0]-costAverages[-1]) / costAverages[0]
        stdreduc = ((costVariances[0]-costVariances[-1]) / costVariances[0] + (sizeVariances[0]-sizeVariances[-1]) / sizeVariances[0]) / 2
        print("In this table, size was reduced by " + str(sizereduc) + "%.")
        print("Cost was reduced by " + str(costreduc) + "%.")
        print("Standard deviation was on average reduced by " + str(stdreduc) + "%.")

    if plot == 0:
        plt.savefig("../results/coverageEvolution.pdf",bbox_inches='tight')
    elif plot == 1:
        plt.savefig("../results/switchesEvolution.pdf")
    plt.show()


def getCITSizeAverage(iterations):
    models = "../data/RIS-FOP/"
    s = SystemData(featuresFile=models + 'features.txt')
    allSize = []
    storage = models + "TestSuitesCIT/testSuite-"
    for i in range(iterations):
        testSuite = computeCITSuite(storage, i, s)
        allSize.append(testSuite.getLength())

    sizeAverage = sum(allSize) / len(allSize)
    sizeVariance = np.std(allSize)
    print("Size average (variance) : " + str(sizeAverage) + " (+-" + str(sizeVariance) + ")")


"""def singleSuite():
    models = "../data/RIS/"
    storage = "../data/RIS/TestSuitesCTT/testSuite/"
    s = SystemData(models+'contexts.txt', models+'features.txt', models+'mapping.txt')
    testSuite = computeCTTSuite(storage, 0, s, recompute=True)
    testSuite.printLatexTransitionForm(mode="unordered")
"""

def getNumberOfSPLOTModels(category=None):
    miniSize = -1
    maxiSize = 101
    if category is not None:
        miniSize = category[0]
        maxiSize = category[1]

    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    #sizes = {}
    total = 0
    for filename in os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        size = len(s.getFeatures())
        if size >= miniSize and size < maxiSize and transitionExist(s):
            total += 1
    return total

def SPLOTmodels(rangeCategory):
    modelFiles = "../data/SPLOT/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-txtconstraints/"
    # rangeCategory = [[10, 12], [12, 15], [15, 20], [20, 25], [25, 30], [30, 40], [40, 51]]
    sizeCategory = [0 for i in rangeCategory]
    total = 0
    for filename in os.listdir(modelFiles):
        total += 1
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        if transitionExist(s):
            found = False
            for cat in range(len(rangeCategory)):
                if rangeCategory[cat][0] <= len(s.getFeatures()) < rangeCategory[cat][1]:
                    sizeCategory[cat] += 1
                    found = True
                    break
            if not found:
                print("didn't find a size category for " + str(filename))
    print(total)
    for r in range(len(rangeCategory)):
        print(str(rangeCategory[r]) + " : " + str(sizeCategory[r]))


def SPLOTimprovements():
    modelFiles = "../data/SPLOT/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-txtconstraints/"
    storageCTT = "../data/SPLOT/SPLOT-TestSuitesCTT/"
    max_iterations = 10
    sizes = [0.0, 0.0, 0.0]
    stds = [0.0, 0.0, 0.0]
    quty = 0

    modes = ["0", "1&2", "1&2&3"]
    for filename in os.listdir(modelFiles):
        quty += 1
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        if transitionExist(s):
            for mode in modes:
                sizeMode = []
                for iteration in range(max_iterations):
                    tempStorageCTT = storageCTT + filename[:-4] + "-" + mode + "-"
                    i_filter, w_la, w_c = recognizeMode(mode)
                    testSuite = computeCTTSuite(tempStorageCTT, iteration, s, 30, i_filter, w_la, w_c)
                    sizeMode.append(testSuite.getLength())
                sizes[modes.index(mode)] += sum(sizeMode) / max_iterations
                stds[modes.index(mode)] += np.std(sizeMode)
    sizes = [s / quty for s in sizes]
    stds = [s / quty for s in stds]
    print("Sizes for modes 0, 12, 123 : " + str(sizes))
    print("Stds for modes 0, 12, 123 : " + str(stds))

def SPLOTgraph():
    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    storageCIT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCIT/"
    storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCTT/"
    max_iterations = 3
    sizesCIT = {}
    sizesCTT = {}
    font = {'size': 16}
    plt.rc('font', **font)

    modes = ["1&2&3"]
    for filename in os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        size = len(s.getFeatures())
        if size >9 and size < 100 and transitionExist(s):
            if size not in sizesCIT:
                sizesCIT[size] = []
                sizesCTT[size] = []
            for iteration in range(max_iterations):
                tempStorageCIT = storageCIT + filename[:-4] + "-"
                testSuite = computeCITSuite(tempStorageCIT, iteration, s, recompute=False)
                # if testSuite.getLength() >= 20:
                #    print(filename)
                sizesCIT[size].append(testSuite.getLength())

                for mode in modes:
                    tempStorageCTT = storageCTT + filename[:-4] + "-" + mode + "-"
                    i_filter, w_la, w_c = recognizeMode(mode)
                    testSuite = computeCTTSuite(tempStorageCTT, iteration, s, 30, i_filter, w_la, w_c, recompute=False)
                    sizesCTT[size].append(testSuite.getLength())
    x = list(sizesCIT.keys())
    x.sort()
    yCIT = []
    yCTT = []
    for size in x:
        yCIT.append(sum(sizesCIT[size]) / len(sizesCIT[size]))
        yCTT.append(sum(sizesCTT[size]) / len(sizesCTT[size]))
    #print(yCIT)
    #print(yCTT)
    print("Correlation coefficient : " + str(scistats.pearsonr(yCIT, yCTT)))
    print("Factor more : " + str(sum(yCTT) / sum(yCIT)))

    x1_smooth, y1_smooth = smoothLinearApprox(x, yCIT)
    x2_smooth, y2_smooth = smoothLinearApprox(x, yCTT)
    x3_smooth, y3_smooth = smoothLinearApprox(x, [yCTT[i]/yCIT[i] for i in range(len(yCIT))])
    #plt.plot(x1_smooth, y1_smooth, '-', label="CIT")
    #plt.plot(x2_smooth, y2_smooth, '--', label="CTT")
    plt.plot(x3_smooth, y3_smooth, '-', label="Coefficient")
    plt.xlabel('Number of features')
    plt.ylabel('Size')
    plt.legend()
    plt.savefig("../results/sizeCTTCIT.pdf")
    plt.show()


def computeCorrelation(sizesCIT, sizesCTT):
    x = list(sizesCIT.keys())
    x.sort()
    yCIT = []
    yCTT = []
    for size in x:
        yCIT.append(sum(sizesCIT[size]) / len(sizesCIT[size]))
        yCTT.append(sum(sizesCTT[size]) / len(sizesCTT[size]))
    # print(yCIT)
    # print(yCTT)
    print("Correlation coefficient : " + str(scistats.pearsonr(yCIT, yCTT)))
    print("Factor more : " + str(sum(yCTT) / sum(yCIT)))

def SPLOTweights(mode, showSize=False, iteration=0, recompute=False, threading=False):
    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    if "3" in mode:
        storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-Comparative/"
        max_iterations = 4
    else:
        storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-Lookahead/"
        max_iterations = 3
    minSize = 10
    maxSize = 50
    rangeCategory = [minSize, maxSize]
    total = getNumberOfSPLOTModels(rangeCategory)
    quty = 0
    print("Computing model " + "0" + "/" + str(total) + " (category: " + str(rangeCategory) + ")", flush=True, end='')
    weightRange = range(0, 11)
    sizes = [0 for i in weightRange]
    costs = [0 for i in weightRange]
    for filename in os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        if minSize <= len(s.getFeatures()) < maxSize and transitionExist(s):
            quty += 1
            with concurrent.futures.ProcessPoolExecutor() as executor:
                print("\rComputing model " + str(quty) + "/" + str(total) + " (category: " + str(
                            rangeCategory) + ") " + str(iteration) + "/"+str(max_iterations), flush=True, end='')
                noComparative=0
                for weight in weightRange:
                    i_filter, w_la, w_c = recognizeMode(mode)
                    if "3" in mode:
                        w_la = 0.7
                        w_c = weight/10
                    else:
                        w_c = 0.5
                        w_la = weight/10

                    currSizes = []
                    currCosts = []
                    suites = []
                    tempThreadsList = []
                    for iteration in range(max_iterations):
                        tempStorageCTT = storageCTT + filename[:-4] + "-" + str(weight/10)
                        sT2 = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
                        if recompute or threading:
                            future = executor.submit(computeCTTSuite, tempStorageCTT, iteration, sT2, interaction_filter=i_filter, weight_lookahead=w_la, weight_comparative=w_c, recompute=recompute)
                            tempThreadsList.append(future)
                        else:
                            suites.append(computeCTTSuite(tempStorageCTT, iteration, sT2, interaction_filter=i_filter, weight_lookahead=w_la, weight_comparative=w_c, recompute=recompute))
                            tempThreadsList.append(0)

                    for i in range(len(tempThreadsList)):
                        if recompute or threading:
                            suites.append(tempThreadsList[i].result())
                        currSizes.append(suites[i].getLength())
                        currCosts.append(suites[i].getCost())
                    if "3" in mode:
                        if weight==0:
                            noComparative = sum(currSizes)/max_iterations
                        #sizes[weight - min(weightRange)] += 1 if sum(currSizes)/max_iterations < noComparative else 0
                        sizes[weight-min(weightRange)] += np.std(currSizes)
                        costs[weight-min(weightRange)] += np.std(currCosts)
                    else:
                        sizes[weight-min(weightRange)] += sum(currSizes)/max_iterations
                        costs[weight-min(weightRange)] += sum(currCosts) / max_iterations

        # create graph
    #sizes = [s/quty*100 for s in sizes[1:]] #[23.6] +
    sizes = [s / quty for s in sizes]
    costs = [s/quty for s in costs]
    fontsize=18
    if showSize:
        plt.plot([w/10 for w in weightRange], sizes)
        #plt.ylabel('Average size of generated test suites', fontsize=fontsize)
        plt.ylabel('Percentage of suites improved by the comparative improvement', fontsize=fontsize)
    else:
        plt.plot([w/10 for w in weightRange], costs)
        plt.ylabel('Average cost of generated test suites', fontsize=fontsize)
    #plt.legend(['interaction coverage', 'transition coverage evolution'])
    # plt.title('Number of switches in a test suite', fontsize=14)
    if "3" in mode:
        plt.xlabel('Comparative weight', fontsize=fontsize)
    else:
        plt.xlabel('Look-ahead weight', fontsize=fontsize)
    plt.show()


def getSPLOTsuites(rangeCategory, modes, max_iterations=3, verbose=False, specificModel=None, recompute=False, threading=True):
    if specificModel is not None:
        rangeCategory = [0, 100]
    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    storageCIT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCIT/"
    storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCTT/"
    minSize = rangeCategory[0]
    maxSize = rangeCategory[1]
    total = getNumberOfSPLOTModels(rangeCategory)
    limitMultiplier = 5
    candidates = 20
    quty=0
    allCITsuites = []
    allCTTsuites = []

    print("Computing model " + "0" + "/" + str(total) + " (category: " + str(rangeCategory) + ")", flush=True, end='')
    for filename in os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        fileCITsuites = []
        fileCTTsuites = [[] for mode in modes]
        threadsCIT = []
        threadsCTT = [[] for mode in modes]

        if specificModel is None or filename == specificModel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                if (minSize <= len(s.getFeatures()) < maxSize and transitionExist(s)) or filename == specificModel:
                    quty += 1
                    if not verbose:
                        print("\rComputing model " + str(quty) + "/" + str(total) + " (category: " + str(
                            rangeCategory) + ")", flush=True, end='')
                    else:
                        print("\rComputing model " + str(quty) + "/" + str(total) + " (category: " + str(
                            rangeCategory) + "), model " + str(filename), flush=True, end='')
                    tempStorageCIT = storageCIT + filename[:-4] + "-"
                    for iteration in range(max_iterations):
                        if False:
                            sT = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
                            future = executor.submit(computeCITSuite, tempStorageCIT, iteration, sT, candidates,recompute=recompute)
                            threadsCIT.append(future)
                        else:
                            fileCITsuites.append(computeCITSuite(tempStorageCIT, iteration, s, candidates, recompute=recompute))

                        tempThreadsList = []
                        tempSuiteList = []
                        if False:
                            fileCITsuites.append(threadsCIT[-1].result())
                        templimit = fileCITsuites[-1].getLength() * limitMultiplier
                        if verbose:
                            print("limit is : ", templimit)

                        for mode in modes:
                            tempStorageCTT = storageCTT + filename[:-4] + "-" + mode + "-"
                            i_filter, w_la, w_c = recognizeMode(mode)
                            # SATSolver.resetCount()
                            if threading:
                                sT2 = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
                                future = executor.submit(computeCTTSuite, tempStorageCTT, iteration, sT2,candidates=candidates,interaction_filter=i_filter, weight_lookahead=w_la,weight_comparative=w_c, recompute=recompute, limit=templimit)
                                tempThreadsList.append(future)
                            else:
                                tempSuiteList.append(
                                    computeCTTSuite(tempStorageCTT, iteration, s, candidates=candidates,
                                                    interaction_filter=i_filter, weight_lookahead=w_la,
                                                    weight_comparative=w_c, recompute=recompute, limit=templimit))
                        if threading:
                            for m in range(len(modes)):
                                threadsCTT[m].append(tempThreadsList[m])
                        else:
                            for m in range(len(modes)):
                                fileCTTsuites[m].append(tempSuiteList[m])
                            # print("Number of SAT calls for CTT, mode "+str(mode) + " : " + str(SATSolver.getCount()))

                    if threading:
                        for m in range(len(modes)):
                            suiteMode = []
                            for t in threadsCTT[m]:
                                suiteMode.append(t.result())
                            fileCTTsuites[m] = suiteMode
                    allCITsuites.append(fileCITsuites)
                    allCTTsuites.append(fileCTTsuites)

    return allCITsuites, allCTTsuites


def getCTTSPLOTresults(rangeCategories, recompute=False, latex=True, onlyCIT=False):
    modes = ["0", "1", "1&2", "1&2&3"]
    max_iterations = 3
    for r in rangeCategories:
        threading = recompute
        allCITsuites, allCTTsuites = getSPLOTsuites(r, modes, max_iterations=max_iterations, threading=threading, recompute=recompute)
        quty = len(allCTTsuites)
        limit = 0
        sizesCIT = sum([t.getLength() for ts in allCITsuites for t in ts])
        #costsCITDiss = sum([t.getCost("dissimilarity") for ts in allCITsuites for t in ts])
        #costsCITCost = sum([t.getCost("minimized") for ts in allCITsuites for t in ts])

        sizesCTT = [0 for i in range(len(modes))]
        costsCTT = [0 for i in range(len(modes))]
        stdSizesCTT = [0 for i in range(len(modes))]
        stdCostsCTT = [0 for i in range(len(modes))]
        NA = [0 for i in range(len(modes))]
        for suites in allCTTsuites:
            for mode in modes:
                suiteMode = suites[modes.index(mode)]
                for testSuite in suiteMode:
                    limit += testSuite.getLimit()
                    if testSuite.isOffLimit():
                        NA[modes.index(mode)] += 1
                    else:
                        sizesCTT[modes.index(mode)] += testSuite.getLength()
                        costsCTT[modes.index(mode)] += testSuite.getCost()
                stdSizesCTT[modes.index(mode)] += np.std([s.getLength() for s in suiteMode])
                stdCostsCTT[modes.index(mode)] += np.std([s.getCost() for s in suiteMode])

        normalise = quty * max_iterations
        limit = round(limit/(normalise*len(modes)), 1)
        stdSizesCTT = [round(s/quty, 1) for s in stdSizesCTT]
        stdCostsCTT = [round(s/quty, 1) for s in stdCostsCTT]
        sizesCTT = [round(sizesCTT[i] / (normalise - NA[i]), 1) if (normalise - NA[i]) > 0 else "N/A" for i in
                    range(len(sizesCTT))]
        costsCTT = [round(costsCTT[i] / (normalise - NA[i]), 1) if (normalise - NA[i]) > 0 else "N/A" for i in
                    range(len(costsCTT))]
        quty = int(quty)
        toPrint = ""
        ranges = str(r[0]) + "-" + str(r[1] - 1)
        NA = [round(na / normalise * 100, 1) for na in NA]

        print("#feature\tQty\t\tSize\t\t\t\t\t\t\t\t\tCost\t\t\t\t\t\t\t\t\t\t\t\t")
        print("\t\t\t\t\tCTT0\tCTT1\tCTT1&2\tCTT1&2&3\tCTT0\tCTT1\tCTT1&2\tCTT1&2&3")
        sizesCTT = [str(sizesCTT[i]) + "$\pm$" + str(stdSizesCTT[i]) + " (" + str(NA[i]) + "%)" for i in range(len(sizesCTT))]
        costsCTT = [str(costsCTT[i]) + "$\pm$" + str(stdCostsCTT[i]) + " (" + str(NA[i]) + "%)" for i in range(len(costsCTT))]
        if latex:
            delimiter = " & "
        else:
            delimiter = "\t"
        for arg in [ranges, quty, limit] + sizesCTT + costsCTT:
            toPrint += str(arg) + delimiter
        toPrint = toPrint.replace("%", "\%")
        if latex:
            print("\r" + toPrint[:-2] + " \\\\\\hline", flush=True)
        else:
            print("\r" + toPrint, flush=True)

        print("#feature\tQty\t\tSize\t\t\t\t\t\t\tCost\t\t\t\t\t\t\t\t\t\tT. cov.")
        print("\t\t\t\t\tCIT\tCTT\tCTT\tCIT Diss\t CIT Cost\tCIT Diss\t CIT Cost")
        sizesCIT = round(sizesCIT/normalise, 1)
        costsCIT = round(costsCIT/normalise, 1)
        #for arg in [ranges, quty, sizesCIT, sizesCTT[-1], costsCITDiss, costsCITCost, costsCTT[-1]]

def SPLOTresults(rangeCategory, recompute=False, computeMetrics=True, latex=False, specificModel = None, verbose=False):
    recomputeCIT = False
    if specificModel is not None:
        rangeCategory = [0, 100]
    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    storageCIT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCIT/"
    storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCTT/"

    #modelFiles = "../data/SPLOT/SPLOT-txt/"
    #constraintsFiles = "../data/SPLOT/SPLOT-txtconstraints/"
    #storageCIT = "../data/SPLOT/SPLOT-TestSuitesCIT/"
    #storageCTT = "../data/SPLOT/SPLOT-TestSuitesCTT/"

    threading = False
    max_iterations = 2
    limitMultiplier = 5

    total = getNumberOfSPLOTModels(rangeCategory)
    minSize = rangeCategory[0]
    maxSize = rangeCategory[1]
    if verbose:
        print("Min size = " + str(minSize) + "; max size = " + str(maxSize))
        print("Total number of models : " + str(total))

    modes = ["0", "1", "1&2", "1&2&3"]
    #modes = ["1&2&3"]
    quty = 0.0
    sizeCIT = 0.0
    sizesCTT = [[] for i in range(len(modes))]
    costCIT = [0, 0]
    costsCTT = [[] for i in range(len(modes))]
    NA = [0 for i in range(len(modes))]

    transitionCoverage = [0, 0]
    candidates = 20
    print("Computing model " + "0" + "/" + str(total) + " (category: " + str(rangeCategory) + ")", flush=True, end='')
    for filename in os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        s = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
        allCITsuites = []
        allCTTsuites = [[] for mode in modes]
        threadsCIT = []
        threadsCTT = [[] for mode in modes]

        if specificModel is None or filename == specificModel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                if (minSize <= len(s.getFeatures()) < maxSize and transitionExist(s)) or filename == specificModel:
                    quty += 1
                    if not verbose:
                        print("\rComputing model " + str(quty) + "/" + str(total) + " (category: " + str(rangeCategory) + ")", flush=True, end='')
                    else:
                        print("\rComputing model " + str(quty) + "/" + str(total) + " (category: " + str(rangeCategory) + "), model " + str(filename), flush=True, end='')

                    if computeMetrics and threading:
                        futureAllTr = executor.submit(allTransitions, s)
                    elif computeMetrics:
                        allTr = allTransitions(s)
                    for iteration in range(max_iterations):
                        tempStorageCIT = storageCIT + filename[:-4] + "-"
                        # SATSolver.resetCount()
                        if threading:
                            sT = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
                            future = executor.submit(computeCITSuite, tempStorageCIT, iteration, sT, candidates,
                                                     recompute=recomputeCIT)
                            threadsCIT.append(future)
                        else:
                            allCITsuites.append(computeCITSuite(tempStorageCIT, iteration, s, candidates, recompute=recomputeCIT))
                        # print("Number of SAT calls for CIT : " + str(SATSolver.getCount()))

                        tempThreadsList = []
                        tempSuiteList = []

                        if threading:
                            allCITsuites.append(threadsCIT[-1].result())

                        templimit = allCITsuites[-1].getLength()*limitMultiplier
                        if verbose:
                            print("limit is : ", templimit)

                        for mode in modes:
                            tempStorageCTT = storageCTT + filename[:-4] + "-" + mode + "-"
                            i_filter, w_la, w_c = recognizeMode(mode)

                            # SATSolver.resetCount()
                            if threading:
                                sT2 = SystemData(featuresFile=txt, extraConstraints=txtConstraints)
                                future = executor.submit(computeCTTSuite, tempStorageCTT, iteration, sT2, candidates=candidates,
                                                    interaction_filter=i_filter, weight_lookahead=w_la,
                                                    weight_comparative=w_c, recompute=recompute, limit=templimit)
                                tempThreadsList.append(future)
                            else:
                                tempSuiteList.append(computeCTTSuite(tempStorageCTT, iteration, s, candidates=candidates,
                                                        interaction_filter=i_filter, weight_lookahead=w_la,
                                                        weight_comparative=w_c, recompute=recompute, limit=templimit))
                        if threading:
                            for m in range(len(modes)):
                                threadsCTT[m].append(tempThreadsList[m])
                        else:
                            for m in range(len(modes)):
                                allCTTsuites[m].append(tempSuiteList[m])
                            # print("Number of SAT calls for CTT, mode "+str(mode) + " : " + str(SATSolver.getCount()))

                    if threading:
                        #for t in threadsCIT:
                        #    allCITsuites.append(t.result())
                        for m in range(len(modes)):
                            suiteMode = []
                            for t in threadsCTT[m]:
                                if verbose:
                                    print("finished mode ", m)
                                suiteMode.append(t.result())
                            allCTTsuites[m] = suiteMode

                    for testSuite in allCITsuites:
                        sizeCIT += testSuite.getLength()
                        costCIT[0] += testSuite.getCost("dissimilarity")
                        costCIT[1] += testSuite.getCost("minimized")
                        if computeMetrics:
                            transitionsDiss = testSuite.transitionPairCoverage("dissimilarity")
                            transitionsMin = testSuite.transitionPairCoverage("minimized")
                            if threading:
                                allTr = futureAllTr.result()
                            transitionCoverage[0] += float(len(transitionsDiss)) / len(allTr) * 100.0
                            transitionCoverage[1] += float(len(transitionsMin)) / len(allTr) * 100.0

                    for mode in modes:
                        suiteMode = allCTTsuites[modes.index(mode)]
                        for testSuite in suiteMode:
                            if testSuite.isOffLimit():
                                NA[modes.index(mode)] += 1
                                if verbose:
                                    print("offlimit for mode ", mode, " at size", testSuite.getLength(), " for model ", filename)
                            else:
                                sizesCTT[modes.index(mode)].append(testSuite.getLength())
                                costsCTT[modes.index(mode)].append(testSuite.getCost())

    normalise = quty * max_iterations
    sizeCIT = round(sizeCIT / normalise, 1)
    #if verbose:
    #    sizesstdCTT = [np.std(s) for s in sizesCTT]
    #    costsstdCTT = [np.std(c) for c in costsCTT]

    sizesCTT = [round(sum(sizesCTT[i]) / (normalise-NA[i]), 1) if (normalise-NA[i]) > 0 else "N/A" for i in range(len(sizesCTT))]
    costCIT = [round(c / normalise, 1) for c in costCIT]
    costsCTT = [round(sum(costsCTT[i]) / (normalise - NA[i]), 1) if (normalise-NA[i]) > 0 else "N/A" for i in range(len(costsCTT))]
    transitionCoverage = [round(t / normalise, 1) for t in transitionCoverage]
    quty = int(quty)
    toPrint = ""
    ranges = str(minSize) + "-" + str(maxSize - 1)
    NA = [round(na/normalise*100, 1) for na in NA]

    sizesCTT = [str(sizesCTT[i]) + "(" + str(NA[i]) + "%)" for i in range(len(sizesCTT))]
    costsCTT = [str(costsCTT[i]) + "(" + str(NA[i]) + "%)" for i in range(len(costsCTT))]
    if latex:
        delimiter = " & "
        #for arg in [ranges, quty, sizeCIT] + sizesCTT + costCIT + costsCTT + transitionCoverage:
        for arg in [ranges, quty, sizeCIT, sizesCTT[-1]] + costCIT + costsCTT[-1:] + transitionCoverage:
            toPrint += str(arg) + delimiter
        toPrint = toPrint.replace("%", "\%")
        print("\r" + toPrint[:-2] + " \\\\\\hline", flush=True)
    else:
        delimiter = "\t\t"
        for arg in [ranges, quty]:
            toPrint += str(arg) + delimiter
        delimiter = "\t"
        for arg in [sizeCIT] + sizesCTT:
            toPrint += str(arg) + delimiter
        toPrint += delimiter
        delimiter = "\t\t"
        for arg in costCIT:
            toPrint += str(arg) + delimiter
        delimiter = "\t"
        for arg in costsCTT:
            toPrint += str(arg) + delimiter
        toPrint += delimiter
        delimiter = "\t\t"
        for arg in transitionCoverage:
            toPrint += str(arg) + delimiter
        print("\r" + toPrint, flush=True)

    #if verbose:
    #    print(sizesstdCTT)
    #    print(costsstdCTT)


def recognizeMode(mode):
    w_la = 0
    w_c = 0
    i_filter = False
    if mode == "0":
        pass
    elif mode == "1":
        i_filter = True
    elif mode == "2":
        w_la = 0.7
    elif mode == "3":
        w_c = 0.5
    elif mode == "1&2" or mode == "1\\&2":
        i_filter = True
        w_la = 0.7
    elif mode == "1&2&3" or mode == "1\\&2\\&3":
        i_filter = True
        w_la = 0.7
        w_c = 0.5
    elif mode == "1&3" or mode == "1\\&3":
        i_filter = True
        w_c = 0.5
    elif mode == "2&3" or mode == "2\\&3":
        w_la = 0.7
        w_c = 0.5
    else:
        print("UNRECOGNIZED MODE")
    return i_filter, w_la, w_c


def findBestWeights(look_ahead=True):
    models = "../data/RIS-FOP/"
    s = SystemData(featuresFile=models + 'features.txt')
    iterations = 100
    values = np.linspace(0, 1, 21)
    sizes = []
    font = {'size': 16}
    plt.rc('font', **font)

    i_filter = True
    w_a = 0.5
    w_c = 0
    for v in values:
        if look_ahead:
            storage = models + "TestSuitesCTT-lookahead/testSuite" + str(round(v, 2)) + "-"
        else:
            storage = models + "TestSuitesCTT-comparative/testSuite" + str(round(v, 2)) + "-"
        allSize = []
        for i in range(iterations):
            if look_ahead:
                w_a = v
            else:
                w_c = v
            testSuite = computeCTTSuite(storage, i, s, 30, interaction_filter=i_filter, weight_lookahead=w_a,
                                        weight_comparative=w_c, recompute=True)
            allSize.append(testSuite.getLength())
        sizes.append(sum(allSize) / len(allSize))
    y1 = sizes
    x1 = values
    print(x1)
    x1_smooth, y1_smooth = smoothLinearApprox(x1, y1)

    plt.plot(x1_smooth, y1_smooth, '-')
    if look_ahead:
        plt.xlabel('Look-ahead weight')
    else:
        plt.xlabel('Comparative weight')
    plt.ylabel('Size')
    if look_ahead:
        plt.savefig("../results/look_ahead.pdf")
    else:
        plt.savefig("../results/comparative.pdf")
    plt.show()


def SPLOTcreateTable(rangeCategories, recompute=False, verbose=False):
    print("#feature\tQty\t\tSize\t\t\t\t\t\t\t\t\tCost\t\t\t\t\t\t\t\t\t\t\t\t\t\tT. cov.")
    print("\t\t\t\t\tCIT\tCTT0\tCTT1\tCTT1&2\tCTT1&2&3\tCIT Diss\tCIT Cost\tCTT0\tCTT1\tCTT1&2\tCTT1&2&3\tCIT Diss\tCIT Cost")
    for r in rangeCategories:
        SPLOTresults(r, recompute=recompute, computeMetrics=True, verbose=verbose, latex=True)

def SPLOTweightsSTD(std=True):
    max_iterations = 3
    sizes = [[] for i in range(1, 11)]
    costs = [[] for i in range(1, 11)]
    for i in range(max_iterations):
        s, c = SPLOTweights("1&2&3", showSize=False, recompute=False, iteration=i, threading=False, returned=True, std=True)
        for i in range(len(sizes)):
            sizes[i].append(s[i])
            costs[i].append(c[i])
    #if std:
    #    sizes = [np.std(s) for s in sizes]
    #    costs = [np.std(c) for c in costs]
    #else:
    sizes = [sum(s)/max_iterations for s in sizes]
    costs = [sum(s) / max_iterations for s in costs]
    plt.plot(np.linspace(0.1, 1, 10), sizes)
    plt.xlabel

    plt.show()

if __name__ == '__main__':
    import cProfile
    #cProfile.run('SPLOTweights("1&2")')
    categories = [[10, 20], [20, 30], [30, 40], [40, 50], [50, 70], [70, 100]]
    #getCTTSPLOTresults([[10, 100]], recompute=False, onlyCIT=True)
    #SPLOTweightsSTD(True)
    #SPLOTweights("1&2&3", threading=False, showSize=True)
    # showAllSPLOTModels()
    #SPLOTmodels()
    #SPLOTcreateTable(categories, recompute=False, verbose=False)
    #SPLOTresults(categories[-1], recompute=True, computeMetrics=False, verbose=True, specificModel="model_20161127_1823059506.txt")
    SPLOTgraph()
    # SPLOTimprovements()
    # getCITcoverage()
    #  ["0", "1", "1&2", "2", "3", "2&3", "1&3", "1&2&3"]
    # RISEvaluation(1000, ["0", "1", "2", "3", "1&2", "2&3", "1&3", "1&2&3"], recompute=False, plot=None)
    # findBestWeights(look_ahead=False)
    # getCITSizeAverage(50)


#model_20130703_1107240012.txt
#model_20161127_1823059506.txt
