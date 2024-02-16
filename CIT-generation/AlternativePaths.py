from OracleSolver import OracleSolver
from TestSuite import allTransitions
import pickle
import os

def computeAlts(fpath, s, testSuite, tag=0, states=4, recompute=False):
    version = "1.0.2"
    filepath = fpath + "-" + str(states) + ".pkl"
    if os.path.exists(filepath) and not recompute:
        alts = readAlts(filepath)
        if not alts.isUpToDate(version):
            alts = AlternativePaths(s, states, version)
    else:
        alts = AlternativePaths(s, states, version)

    if alts.computedTestSuite(tag):
        return alts.altPathsForTestSuite(testSuite, tag=tag)
    else:
        toReturn = alts.altPathsForTestSuite(testSuite, tag=tag)
        storeAlts(alts, filepath)
        return toReturn

def readAlts(filePath):
    alts = pickle.load(open(filePath, 'rb'))
    return alts

def storeAlts(alts, filePath):
    f = open(filePath, "wb")
    pickle.dump(alts, f)
    f.close()

class AlternativePaths:
    def __init__(self, systemData, states=4, version="1.0.0", verbose=False):
        self.version = version
        self.verbose = verbose
        self.s = systemData
        self.states = states
        self.allTransitions = allTransitions(self.s)
        self.decomposableTransitions = None
        self.nonDecomposableTransitions = None
        #self.decomposableTransitions, self.nonDecomposableTransitions = self.solver.preprocessTransitions(self.allTransitions)
        self.allResults = {}

    def isUpToDate(self, version):
        if hasattr(self, "version"):
           return self.version == version
        return False

    def computedTestSuite(self, tag):
        return tag in self.allResults

    def altPathsForTestSuite(self, testSuite, tag=0):
        if tag in self.allResults:
            return self.allResults[tag][0], self.allResults[tag][1]
        solver = OracleSolver(self.s, self.states)
        transitionsToCover = self.allTransitions
        totalTransitions = len(transitionsToCover)
        allUncoverablesTransitions = []
        allPaths = []
        if self.decomposableTransitions is not None:
            transitionsToCover = self.decomposableTransitions
        transitionsToCover = [self.simplifiedTransition(t) for t in transitionsToCover]
        for i in range(len(testSuite)-1):
            #if len(transitionsToCover) == 0:
            #    print("Transition complete before end of test suite is abnormal. Error.")
            #    return None
            if self.verbose:
                print("commencing test number ", i)
            newPaths, transitionsToCover, uncoverableTransitions = self.createAlternativePaths(testSuite[i], testSuite[i+1], transitionsToCover, solver)
            allUncoverablesTransitions = allUncoverablesTransitions + uncoverableTransitions
            allPaths.append(newPaths)

        undetectables = len(allUncoverablesTransitions)/totalTransitions
        self.allResults[tag] = allPaths, undetectables
        return allPaths, undetectables

    def createAlternativePaths(self, config1, config2, transitionsToCover, solver, prevUncoverables = []):
        #algo for branching off paths
        changes = [(f, config2[f]) for f in config1 if config1[f] != config2[f]]
        possibleCoverage = []
        for i in range(len(changes)-1):
            for j in range(i+1, len(changes)):
                t = self.simplifiedTransition((changes[i], changes[j]))
                if t in transitionsToCover:
                    possibleCoverage.append(t)
        if len(possibleCoverage) == 0:
            return [], transitionsToCover, []
        possiblePathCoverage = [possibleCoverage]
        currentCoverage = []
        uncoverableTransitions = []
        paths = []
        while len(possiblePathCoverage) > 0:
            possibleCoverage = possiblePathCoverage.pop()
            path = solver.createPath(config1, config2, possibleCoverage)
            if path is None:
                if len(possibleCoverage) > 1:
                    possiblePathCoverage.append(possibleCoverage[:round(len(possibleCoverage)/2)])
                    possiblePathCoverage.append(possibleCoverage[round(len(possibleCoverage)/2):])
                else:
                    uncoverableTransitions.append(possibleCoverage[0])
                    transitionsToCover.remove(possibleCoverage[0])
            else:
                paths.append(path)
                for t in possibleCoverage:
                    currentCoverage.append(t)
        if len(uncoverableTransitions) == 0:
            for t in currentCoverage:
                transitionsToCover.remove(t)
            return paths, transitionsToCover, prevUncoverables
        else:
            return self.createAlternativePaths(config1, config2, transitionsToCover, solver, prevUncoverables=prevUncoverables+uncoverableTransitions)

    def getNondecomposableTransitions(self):
        return self.nonDecomposableTransitions.copy()

    def getDecomposableTransitions(self):
        return self.decomposableTransitions.copy()

    def simplifiedTransition(self, transition):
        orderedTransition = transition
        if transition[0][0] < transition[1][0]:
            orderedTransition = (transition[1], transition[0])
        value0 = 1 if orderedTransition[0][1] > 0 else -1
        value1 = 1 if orderedTransition[1][1] > 0 else -1
        simplifiedTransition = ((orderedTransition[0][0], value0), (orderedTransition[1][0], value1))
        return simplifiedTransition
