from CITSAT import CITSAT
from ResultRefining import printCoveringArray, numberOfChangements, orderArray, addedCreationCost
from SystemData import SystemData
import time
from pysat.solvers import Glucose3
from TestsEvolution import TestsEvolution
from SATSolver import SATSolver
import math
import numpy as np
from numpy import polyval, polyfit
# was scipy instead of numpy
import matplotlib.pyplot as plt
"""Uses CITSAT() and displays the results.
"""
def testCITSATData():
    s = SystemData('contexts.txt', 'features.txt', 'mapping.txt')
    result = CITSAT(s, False)
    printCoveringArray(result, s, "Normal")
    print("================================REFINED MODE=====================================")
    printCoveringArray(result, s, "Refined")

def singleRun():
    time1 = time.time()
    models = "./data/enlarged/"
    s = SystemData(models+'contexts.txt', models+'features.txt', models+'mapping.txt')
    result = CITSAT(s, False, 30)
    totalTime = time.time() - time1
    #printCoveringArray(result, s, "Normal", order=False)
    print("================================ORDER = False=====================================")
    printCoveringArray(result, s, "Refined", writeMode=False, order=False)
    print("================================ORDER = True=====================================")
    printCoveringArray(result, s, "Refined", writeMode=False, order=True)
    print("Computation time : " + str(totalTime) + " seconds")
    unrefinedCost = numberOfChangements(result, s.getContexts())
    print("COST UNREFINED : " + str(unrefinedCost))
    refinedCost = numberOfChangements(orderArray(result, s.getContexts()), s.getContexts())
    print("COST REFINED : " + str(refinedCost))
    print("Decrease in cost of : " + str((unrefinedCost - refinedCost)/unrefinedCost))


def multipleRuns(iterations):
    maxTime = 0
    minTime = 10000
    sumTime = 0
    iterations = 10
    for i in range(iterations):
        time1 = time.process_time()
        testCITSATData()
        totalTime = time.process_time() - time1
        maxTime = max(totalTime, maxTime)
        minTime = min(totalTime, minTime)
        sumTime += totalTime
        print(str(i) + "nth iteration. Computation took " + str(totalTime) + " seconds.")

    print("We took " + str(sumTime) + " seconds to compute " + str(iterations) + ".")
    print("The maximum time is " + str(maxTime) + " seconds.")
    print("The minimum time is " + str(minTime) + " seconds.")

def thesisExample():
    example = {}
    example['Brand'] = ['Nokia', 'Samsung']
    example['State'] = ['Broken', 'Used', 'New']
    example['Performance'] = ['Slow', 'Fast']
    result = CITSAT(None, example, False)
    for testCase in result:
        print(testCase)


def incrementalRun(mode="SAT", verbose=True):
    models = ["./data/minimalist/", "./data/normal_size/", "./data/enlarged/"]

    s1 = SystemData(models[0] + 'contexts.txt', models[0] + 'features.txt', models[0] + 'mapping.txt')
    s2 = SystemData(models[1] + 'contexts.txt', models[1] + 'features.txt', models[1] + 'mapping.txt')
    s3 = SystemData(models[2] + 'contexts.txt', models[2] + 'features.txt', models[2] + 'mapping.txt')

    result1 = CITSAT(s1)
    if mode is "SAT":
        Feat1to2 = TestsEvolution([s1.getNodes(), result1], s2, mode)
    else:
        Feat1to2 = TestsEvolution([s1.getNodes(), result1], s2, 6)
    result2 = CITSAT(s2, False, 30, Feat1to2)
    Feat2to3 = TestsEvolution([s2.getNodes(), result2], s3, mode)
    result3 = CITSAT(s3, False, 30, Feat2to3)

    #printCoveringArray(result3, s3, "Refined", evolution=Feat2to3)
    #printCoveringArray(result3, s3, mode="Refined", evolution=Feat2to3)
    prevResult = result2
    prevSystem = s2
    nextSystem = s3
    nextResult = result3
    change = Feat2to3
    modifCost = numberOfChangements(nextResult[:len(prevResult)], nextSystem.getContexts(), change.getNewNodes())

    newCreationCost = addedCreationCost(nextResult, nextSystem.getContexts(), change.getNumberPrevTests())
    if verbose:
        print("Modif : " + str(modifCost) + " - new creation : " + str(newCreationCost) + " - Total : " + str(modifCost+newCreationCost))
        print("Added test cases : " + str(len(nextResult) - len(prevResult)))
        print("Real steps taken : " + str(change.getRealSteps()))
        print("--------------")
    return [modifCost, (modifCost+newCreationCost), len(nextResult) - len(prevResult), change.getRealSteps()]


def myLittleTests():
    g = Glucose3(incr=True)
    g.add_clause([1, 2], [3, 4])
    print(g.propagate([-1, 3]))


def anotherTest():
    models = "./data/enlarged/"
    s = SystemData(models + 'contexts.txt', models + 'features.txt', models + 'mapping.txt')
    mySat = SATSolver(s)
    (_, propagated) = mySat.propagate([s.toIndex("AddSystem")])
    for p in propagated:
        print("Propagated node : " + str(s.getNodes()[abs(p)]))


def rearrangementMetricsTest(iterations):
    models = ["./data/minimalist/", "./data/normal_size/", "./data/enlarged/"]
    models = [models[2]]
    for model in models:
        unrefinedScore = 0
        refinedScore = 0
        unrefinedFeatureScore = 0
        refinedFeatureScore = 0
        increaseScore = 0
        tmpUnrefinedVariance = 0
        tmpRefinedVariance = 0
        tmpIncreaseVariance = 0
        sizeScore = 0
        timeScore = 0
        propagations = 0
        propagatedScore = 0
        for i in range(iterations):
            s = SystemData(model + 'contexts.txt', model + 'features.txt', model + 'mapping.txt')
            veryUglyWay = []
            time1 = time.time()
            result = CITSAT(s, False, 30, veryUglyWay=veryUglyWay)
            timeScore += time.time() - time1
            if len(veryUglyWay) > 0:
                propagations += veryUglyWay[0][0]
                propagatedScore += veryUglyWay[0][1]
            sizeScore += len(result)
            unrefined = numberOfChangements(result, s.getContexts())
            unrefinedScore += unrefined
            unrefinedFeatureScore += numberOfChangements(result, s.getFeatures())
            tmpUnrefinedVariance += unrefined * unrefined
            refined = numberOfChangements(orderArray(result, s.getContexts()), s.getContexts())
            refinedScore += refined
            refinedFeatureScore += numberOfChangements(orderArray(result, s.getContexts()), s.getFeatures())
            tmpRefinedVariance += refined * refined
            increase = 100*(unrefined - refined) / unrefined
            increaseScore += increase
            tmpIncreaseVariance += increase * increase
        print("Average size of the arrays : " + str(sizeScore/iterations))
        print("Unrefined score : " + str(unrefinedScore) + "; in average : " + str(unrefinedScore/iterations))
        print("Refined score : " + str(refinedScore) + "; in average : " + str(refinedScore/iterations))
        print("(From the score themselves) Average decrease of " + str(100*(unrefinedScore - refinedScore)/unrefinedScore) + " %")
        print("Average decrease score of : " + str(increaseScore/iterations) + " %")
        print("Variance of the score : " + str(math.sqrt(tmpIncreaseVariance/iterations - ((increaseScore/iterations) * (increaseScore/iterations)))))
        print("Average of time taken : " + str(timeScore/iterations))
        print("Average propagations : " + str(propagations/iterations) + " - Average propagated nodes : " + str(propagatedScore/iterations))
        print("Unrefined FEATURE score : " + str(unrefinedFeatureScore) + "; in average : " + str(unrefinedFeatureScore/iterations))
        print("Refined FEATURE score : " + str(refinedFeatureScore) + "; in average : " + str(refinedFeatureScore/iterations))
        print("(From the score themselves) FEATURE SCORE Average decrease of " + str(100*(unrefinedFeatureScore - refinedFeatureScore)/unrefinedFeatureScore) + " %")


def evolutionMetrics(iterations):
    models = ["./data/minimalist/", "./data/normal_size/", "./data/enlarged/"]
    cases = ["SAT1to2", "SAT2to3", "Feat1to2", "Feat2to3", "Gen2", "Gen3"]
    modificationCost = {}
    newCreationCost = {}
    totalCost = {}
    sizeScore = {}
    for case in cases:
        modificationCost[case] = 0
        newCreationCost[case] = 0
        totalCost[case] = 0
        sizeScore[case] = 0

    for i in range(iterations):
        s1 = SystemData(models[0] + 'contexts.txt', models[0] + 'features.txt', models[0] + 'mapping.txt')
        s2 = SystemData(models[1] + 'contexts.txt', models[1] + 'features.txt', models[1] + 'mapping.txt')
        s3 = SystemData(models[2] + 'contexts.txt', models[2] + 'features.txt', models[2] + 'mapping.txt')

        result1 = CITSAT(s1, False, 30)
        #result2 = CITSAT(s2, False, 30)
        #result3 = CITSAT(s3, False, 30)

        #newCreationCost["Gen2"] = newCreationCost["Gen2"] + numberOfChangements(result2, s2.getContexts())
        #newCreationCost["Gen3"] = newCreationCost["Gen3"] + numberOfChangements(result3, s3.getContexts())
        #sizeScore["Gen2"] = sizeScore["Gen2"] + len(result2)
        #sizeScore["Gen3"] = sizeScore["Gen3"] + len(result3)
        """
        SAT1to2 = TestsEvolution([s1.getNodes(), result1], s2, mode="SAT")
        result2SAT = CITSAT(s2, False, 30, testsEvolution=SAT1to2)
        SAT2to3 = TestsEvolution([s2.getNodes(), result2SAT], s3, mode="SAT")
        result3SAT = CITSAT(s3, False, 30, testsEvolution=SAT2to3)

        modificationCost["SAT1to2"] = modificationCost["SAT1to2"] + numberOfChangements(result2SAT[:len(result1)], s2.getContexts(), SAT1to2.newNodes)
        modificationCost["SAT2to3"] = modificationCost["SAT2to3"] + numberOfChangements(result3SAT[:len(result2SAT)], s3.getContexts(), SAT2to3.newNodes)
        newCreationCost["SAT1to2"] = newCreationCost["SAT1to2"] + addedCreationCost(result2SAT, s2.getContexts(), SAT1to2.getNumberPrevTests())
        newCreationCost["SAT2to3"] = newCreationCost["SAT2to3"] + addedCreationCost(result3SAT, s3.getContexts(), SAT2to3.getNumberPrevTests())
        sizeScore["SAT1to2"] = sizeScore["SAT1to2"] + len(result2SAT)
        sizeScore["SAT2to3"] = sizeScore["SAT2to3"] + len(result3SAT)"""

        Feat1to2 = TestsEvolution([s1.getNodes(), result1], s2, mode=4)
        result2Feat = CITSAT(s2, False, 30, testsEvolution=Feat1to2)
        Feat2to3 = TestsEvolution([s2.getNodes(), result2Feat], s3, mode=8)
        result3Feat = CITSAT(s3, False, 30, testsEvolution=Feat2to3)

        modificationCost["Feat1to2"] = modificationCost["Feat1to2"] + numberOfChangements(result2Feat[:len(result1)], s2.getContexts(), Feat1to2.newNodes)
        modificationCost["Feat2to3"] = modificationCost["Feat2to3"] + numberOfChangements(result3Feat[:len(result2Feat)], s3.getContexts(), Feat2to3.newNodes)
        newCreationCost["Feat1to2"] = newCreationCost["Feat1to2"] + addedCreationCost(result2Feat, s2.getContexts(), Feat1to2.getNumberPrevTests())
        newCreationCost["Feat2to3"] = newCreationCost["Feat2to3"] + addedCreationCost(result3Feat, s3.getContexts(), Feat2to3.getNumberPrevTests())
        sizeScore["Feat1to2"] = sizeScore["Feat1to2"] + len(result2Feat)
        sizeScore["Feat2to3"] = sizeScore["Feat2to3"] + len(result3Feat)

    print("MODIFICATION COST - NEW TEST COST - TOTAL COST - SIZE")
    for case in cases:
        print("FOR CASE : " + case)
        newLine = str(modificationCost[case] / iterations) + " - "
        newLine += str(newCreationCost[case] / iterations) + " - "
        newLine += str(totalCost[case] / iterations) + " - "
        newLine += str(sizeScore[case] / iterations)
        print(newLine)


def procedureForIncrementalTesting():
    modifCost = []
    totalCosts = []
    newTests = []
    realSteps = []
    repeats = 10
    for i in range(10, 13):
        tmpModifCost = 0
        tmptotalCost = 0
        tmpNewTest = 0
        tmpStep = 0
        for iter in range(repeats):
            info = incrementalRun(i, False)
            tmpModifCost += info[0]
            tmptotalCost += info[1]
            tmpNewTest += info[2]
            tmpStep += info[3]
        totalCosts.append(tmptotalCost / repeats)
        newTests.append(tmpNewTest / repeats)
        realSteps.append(tmpStep / repeats)
        modifCost.append(tmpModifCost / repeats)
        print("For S = " + str(i) + ", cost : " + str(tmptotalCost / repeats) + ", modif cost : " + str(tmpModifCost / repeats) +  ", newTests : " + str(
            tmpNewTest / repeats) + ", real steps : " + str(tmpStep / repeats))
    print("Total costs : " + str(totalCosts))
    print("Added tests : " + str(newTests))
    print("Real steps : " + str(realSteps))

def plotAnalysisOfS():
    y = [59.3, 46.2, 38, 37.5, 37.5, 39.0, 39.4, 35.7, 35.8, 35, 37.4, 38.2, 39.7, 42, 43.5]
    x = np.linspace(1, len(y)+1, num=len(y), endpoint=True)
    #f = interp1d(x, y, kind='cubic')
    xnew = np.linspace(1, len(y)+1, num=41, endpoint=True)
    param = polyfit(x, y, 2)
    f = polyval(param, xnew)

    font = {'family': 'normal',
            'size': 16}
    plt.rc('font', **font)

    plt.plot(x, y, 'o', xnew, f, '--')
    plt.legend(['Data', 'Least square approximation'])
    plt.title('')
    plt.xlabel('S')
    plt.ylabel('Total cost')
    #test = [1, 3, 5, 7, 9, 11, 13, 15]
    #plt.xticks(test, test)
    plt.savefig("results/totalCost.pdf")
    plt.show()

    y = [5.0, 5.2, 4.6, 5.3, 5.7, 5.2, 6.1, 6.0, 5.9, 5.5, 6.4, 6.5, 7.2, 7.4, 7.4]
    x = np.linspace(1, len(y) + 1, num=len(y), endpoint=True)
    xnew = np.linspace(1, len(y) + 1, num=41, endpoint=True)
    param = polyfit(x, y, 2)
    f = polyval(param, xnew)
    plt.plot(x, y, 'o', xnew, f, '--')
    plt.legend(['Data', 'Least square approximation'])
    plt.title('')
    plt.xlabel('S')
    #test = [1, 3, 5, 7, 9, 11, 13, 15]
    #plt.xticks(test, test)
    plt.ylabel('Number of new tests')
    plt.savefig("results/addedTests.pdf")
    plt.show()

    y = [1.0, 1.8, 2.4, 2.9, 3.3, 3.8, 4.3, 4.325, 4.7, 5.5, 4.9, 6, 6.2, 5.4]
    x = np.linspace(1, len(y) + 1, num=len(y), endpoint=True)
    xnew = np.linspace(1, len(y) + 1, num=41, endpoint=True)
    param = polyfit(x, y, 2)
    f = polyval(param, xnew)
    plt.plot(x, y, 'o', xnew, f, '--', x, x, '-')
    plt.legend(['Data', 'Least square approximation', 'Theoretical'])
    plt.title('')
    plt.xlabel('S')
    plt.ylabel('updates/partial scenario')
    #test = [1, 3, 5, 7, 9, 11, 13, 15]
    #plt.xticks(test, test)
    plt.savefig("results/testcase-scenario.pdf")
    plt.show()


# evolutionMetrics(10)
# rearrangementMetricsTest(20)
# anotherTest()

# singleRun() # Runs a single time the algorithm and displays the results

# procedureForIncrementalTesting()

plotAnalysisOfS()

# multipleRuns(3)

# thesisExample()


