from SATSolver import SATSolver
import random

class BuildingCTT:
    def __init__(self, systemData, verbose=False, numCandidates=30, interaction_filter=True, weight_lookahead=0.5, weight_comparative=0.3, limit=1000):
        self.verbose = verbose
        self.limit = limit
        self.stopgapmeasure = False
        self.bestOfBoth = False
        self.numCandidates = numCandidates
        self.systemData = systemData
        self.solver = SATSolver(systemData)
        self.valuesForFactors = systemData.getValuesForFactors()
        self.core = self.discoverCore()
        for key in self.core.keys():
            del self.valuesForFactors[key]

        self.prevWasDecidedByLookahead = False
        self.base_weight_lookahead = weight_lookahead
        self.weight_lookahead = weight_lookahead
        self.weight_comparative = weight_comparative
        self.interaction_filter = interaction_filter

        self.coveringArray = []
        self.numTests = 0
        self.unCovSets, self.unCovTransitions, self.unCovPairsCount = self.computeSetToCover()
        if verbose:
            print("Number of uncovered interactions and transitions : " + str(len(self.unCovSets)) + " - " + str(len(self.unCovTransitions)))
        self.totalNumberOfPairs = len(self.unCovSets) + len(self.unCovTransitions)
        # self.unCovSets, self.unCovTransitions, self.unCovPairsCount = self.computeSetToCover("basic")
        self.factors = list(self.valuesForFactors.keys())

    def computeScores(self, prevTestCase):
        tempUnCovPairsCnt = self.unCovPairsCount.copy()
        for transition in self.unCovTransitions:
            # if transition cannot be covered.
            if prevTestCase is None or transition[0][1] == prevTestCase[transition[0][0]] or transition[1][1] == prevTestCase[transition[1][0]]:
                tempUnCovPairsCnt[transition[0]] = tempUnCovPairsCnt[transition[0]] - 1
                tempUnCovPairsCnt[transition[1]] = tempUnCovPairsCnt[transition[1]] - 1

                if prevTestCase is not None:
                    oppositePair = (transition[0][0], -1*transition[0][1])
                    tempUnCovPairsCnt[oppositePair] = tempUnCovPairsCnt[oppositePair] + self.weight_lookahead
                    oppositePair = (transition[1][0], -1*transition[1][1])
                    tempUnCovPairsCnt[oppositePair] = tempUnCovPairsCnt[oppositePair] + self.weight_lookahead
                # Following is not logical.
                """if prevTestCase is not None and transition[0][1] == prevTestCase[transition[0][0]]:
                    oppositePair = (transition[0][0], -1*transition[0][1])
                    tempUnCovPairsCnt[oppositePair] = tempUnCovPairsCnt[oppositePair] + self.lookahead_weight
                if prevTestCase is not None and transition[1][1] == prevTestCase[transition[1][0]]:
                    oppositePair = (transition[1][0], -1*transition[1][1])
                    tempUnCovPairsCnt[oppositePair] = tempUnCovPairsCnt[oppositePair] + self.lookahead_weight"""

        # INTERACTIONFILTER IS ALREADY COMPUTED WHEN COMPUTING SET TO COVER.

        # Comparative scores.
        reTempUnCovPairsCnt = {}
        for pair in tempUnCovPairsCnt:
            #tempUnCovPairsCnt[pair] = tempUnCovPairsCnt[pair] - self.weight_comparative*tempUnCovPairsCnt[(pair[0], -pair[1])]
            reTempUnCovPairsCnt[pair] = tempUnCovPairsCnt[pair] - self.weight_comparative*tempUnCovPairsCnt[(pair[0], -pair[1])]

        return reTempUnCovPairsCnt

    def selectSpecificBestValue(self, f, currentTestCase, prevTestCase):
        candidates = []
        bestScore = -1
        scores = []
        values = self.valuesForFactors[f]
        lostscore = 0

        for v in values:
            pair = (f, v)
            score = 0

            possibleInteractions = [self.orderedSet([pair, (testFactor, currentTestCase[testFactor])])
                                    for testFactor in currentTestCase]
            possibleTransitions = []
            if prevTestCase is not None and prevTestCase[f] != v:
                possibleTransitions = [self.orderedSet([pair, (otherF, currentTestCase[otherF])])
                                       for otherF in currentTestCase if currentTestCase[otherF] != prevTestCase[otherF]]

            for interaction in self.unCovSets:
                if interaction in possibleInteractions:
                    score += 1

            for transition in self.unCovTransitions:
                if transition in possibleTransitions:
                    score += 1

            if self.weight_lookahead > 0:
                futureTestCase = [-v]
                shuffledUncovs = self.unCovTransitions.copy()
                random.shuffle(shuffledUncovs)
                for transition in shuffledUncovs:
                    if transition not in possibleTransitions:
                        # LOOKAHEAD SCORES IF WE PREPARE FOR A FUTURE TRANSITION BY USING THIS VALUE.
                        if transition[0] == (f, -v) and transition[1][0] in currentTestCase and currentTestCase[transition[1][0]] == -1 * transition[1][1]:

                            if transition[1][1] in futureTestCase or self.solver.checkSAT(futureTestCase + [transition[1][1]]):
                                score += self.weight_lookahead
                                futureTestCase = futureTestCase + [transition[1][1]] if transition[1][1] not in futureTestCase else futureTestCase
                                futureTestCase = futureTestCase + [transition[0][1]] if transition[0][1] not in futureTestCase else futureTestCase

                        condition2 = transition[1] == (f, -v)
                        if transition[1] == (f, -v) and transition[0][0] in currentTestCase and currentTestCase[transition[0][0]] == -1 * transition[0][1]:
                            if transition[0][1] in futureTestCase or self.solver.checkSAT(futureTestCase + [transition[0][1]]):
                                score += self.weight_lookahead
                                futureTestCase = futureTestCase + [transition[0][1]] if transition[0][1] not in futureTestCase else futureTestCase
                                futureTestCase = futureTestCase + [transition[1][1]] if transition[1][1] not in futureTestCase else futureTestCase

            scores.append(score)
        for score in scores:
            if score > bestScore:
                bestScore = score
                candidates = [values[scores.index(score)]]
            elif score == bestScore:
                candidates.append(values[scores.index(score)])
        return random.choice(candidates)

    def selectBestTestCase(self, testCasePool, prevTestCase):
        candidates = []
        bestScore = -1
        bestLookaheadscore = -1
        bestCost = None
        scores = []
        lookaheads = []
        costs = []
        for testCase in testCasePool:
            possibleInteractions = self.computeAllPairs(testCase)
            possibleTransitions = []
            #cost = None
            if prevTestCase is not None:
                #cost = sum([1 for v in testCase.values() if v not in prevTestCase.values()])
                possibleTransitions = [pair for pair in possibleInteractions if prevTestCase[pair[0][0]] != pair[0][1] and prevTestCase[pair[1][0]] != pair[1][1]]

            # ADD HEURISTICS HERE.
            score = 0
            lookaheadscore = 0
            for set in possibleInteractions:
                if set in self.unCovSets:
                    score += 1

            futureTestCase = []
            lostscore = 0
            shuffledUncovs = self.unCovTransitions.copy()
            random.shuffle(shuffledUncovs)
            for transition in shuffledUncovs:
                if transition in possibleTransitions:
                    score += 1
                else:
                    # LOOKAHEAD SCORES IF WE PREPARE FOR A FUTURE TRANSITION BY USING THIS TEST CASE.
                    if testCase[transition[0][0]] == -transition[0][1] and testCase[transition[1][0]] == -transition[1][1]:
                        tempFutureCase = futureTestCase + [t for t in [transition[0][1], transition[1][1]] if t not in futureTestCase]
                        if self.solver.checkSAT(tempFutureCase):
                            futureTestCase = tempFutureCase
                            lookaheadscore += self.weight_lookahead
                        else:
                            lostscore += self.weight_lookahead
            #print("scoring score is ", score, "lookaheadscore is ", lookaheadscore, "lost score is ", lostscore)

            scores.append(score)
            lookaheads.append(lookaheadscore)
            #costs.append(cost)
            cost_influence = 0
            #if self.weight_cost is not None and bestCost is not None and cost is not None and cost != 0 and bestCost != 0:
            #    cost_influence = (cost - bestCost)/bestCost*self.weight_cost
            if (score+lookaheadscore)*(1-cost_influence) > bestScore+bestLookaheadscore:
                bestScore = score
                bestLookaheadscore = lookaheadscore
                #bestCost = cost
                # print("Best score is " + str(bestScore) + ",  look ahead : " + str(lookaheadscore))
                candidates = [testCase]
            elif score+lookaheadscore == bestScore+bestLookaheadscore:
                candidates.append(testCase)
        """print("----------")
        print(scores)
        print(lookaheads)
        print("----------")"""
        #print(bestLookaheadscore, bestScore)
        if self.stopgapmeasure and bestLookaheadscore > bestScore:
            self.prevWasDecidedByLookahead = True
        return random.choice(candidates)

    def updateUnCovSets(self, testCase, prevTestCase):
        possibleInteractions = self.computeAllPairs(testCase)
        possibleTransitions = []
        improvedCoverage = False
        if prevTestCase is not None:
            possibleTransitions = [pair for pair in possibleInteractions if prevTestCase[pair[0][0]] != pair[0][1] and prevTestCase[pair[1][0]] != pair[1][1]]
        for set in possibleInteractions:
            if set in self.unCovSets:
                improvedCoverage = True
                self.unCovSets.remove(set)
                self.unCovPairsCount[set[0]] = self.unCovPairsCount[set[0]] - 1
                self.unCovPairsCount[set[1]] = self.unCovPairsCount[set[1]] - 1
                if self.unCovPairsCount[set[0]] < 0 or self.unCovPairsCount[set[1]] < 0:
                    print("PROBLEM IN UPDATEUNCOVSETS; SETS")

        for transition in possibleTransitions:
            if transition in self.unCovTransitions:
                improvedCoverage = True
                self.unCovTransitions.remove(transition)
                self.unCovPairsCount[transition[0]] = self.unCovPairsCount[transition[0]] - 1
                self.unCovPairsCount[transition[1]] = self.unCovPairsCount[transition[1]] - 1
                if self.unCovPairsCount[transition[0]] < 0 or self.unCovPairsCount[transition[1]] < 0:
                    print("PROBLEM IN UPDATEUNCOVSETS; TRANSITIONS")
        return improvedCoverage

    def computeAllPairs(self, testCase):
        possibleSets = []
        for i in range(len(self.factors)-1):
            pair1 = (self.factors[i], testCase[self.factors[i]])
            for j in range(i+1, len(self.factors)):
                pair2 = (self.factors[j], testCase[self.factors[j]])
                possibleSets.append([pair1, pair2])
        return possibleSets

    """A set is under the form ((feature1, _), (feature2, _)).
    orderedSet returns the set with feature1 and feature2 ordered like in the keys of valuesForFactors.
    """
    def orderedSet(self, interaction):
        if self.factors.index(interaction[0][0]) < self.factors.index(interaction[1][0]):
            return interaction
        else:
            return [interaction[1], interaction[0]]

    def propagateCurrentTestCase(self, newTestCase):
        (_, newValues) = self.solver.propagate(newTestCase.values())
        finalNodes = self.systemData.getNodes()
        for value in newValues:
            newTestCase[finalNodes[abs(value)]] = value
        return newTestCase

    def discoverCore(self):
        core = {}
        for factor in self.valuesForFactors:
            if not self.solver.checkSAT([self.valuesForFactors[factor][1]]):
                core[factor] = self.valuesForFactors[factor][0]
            if not self.solver.checkSAT([self.valuesForFactors[factor][0]]):
                core[factor] = self.valuesForFactors[factor][1]
        return core

    """Computes every possible pair in valuesForFactors, but removes those not compatible with the SAT solver"""
    def computeSetToCover(self):
        unCovSets = []
        unCovTransitions = []
        unCovPairsCount = {}
        factors = list(self.valuesForFactors.keys())
        for i in range(len(factors) - 1):
            for j in range(len(self.valuesForFactors[factors[i]])):
                pair1 = (factors[i], self.valuesForFactors[factors[i]][j])
                for i2 in range(i + 1, len(factors)):
                    for j2 in range(len(self.valuesForFactors[factors[i2]])):
                        pair2 = (factors[i2], self.valuesForFactors[factors[i2]][j2])

                        if self.solver.checkSAT([pair1[1], pair2[1]]):
                            # Each set is a [Factor, Value, Factor, Value, ...] tuple
                            if not self.interaction_filter:
                                unCovSets.append([pair1, pair2])
                                if pair1 not in unCovPairsCount:
                                    unCovPairsCount[pair1] = 1.0
                                else:
                                    unCovPairsCount[pair1] += 1.0
                                if pair2 not in unCovPairsCount:
                                    unCovPairsCount[pair2] = 1.0
                                else:
                                    unCovPairsCount[pair2] += 1.0

                                if self.solver.checkSAT([-1*pair1[1], -1*pair2[1]]):
                                    unCovTransitions.append([pair1, pair2])
                                    unCovPairsCount[pair1] += 1
                                    unCovPairsCount[pair2] += 1
                            elif self.interaction_filter:
                                if pair1 not in unCovPairsCount:
                                    unCovPairsCount[pair1] = 1
                                else:
                                    unCovPairsCount[pair1] += 1
                                if pair2 not in unCovPairsCount:
                                    unCovPairsCount[pair2] = 1
                                else:
                                    unCovPairsCount[pair2] += 1

                                if self.solver.checkSAT([-1*pair1[1], -1*pair2[1]]):
                                    unCovTransitions.append([pair1, pair2])
                                else:
                                    unCovSets.append([pair1, pair2])
                            else:
                                print("DID NOT RECOGNIZE HEURISTICS IN COMPUTESETTOCOVER.")
        return unCovSets, unCovTransitions, unCovPairsCount

    def getCoveringArray(self):
        if len(self.valuesForFactors) == 0:
            return [self.core]

        if len(self.valuesForFactors) == 1:
            dupl1 = self.core.copy()
            dupl2 = self.core.copy()
            key = list(self.valuesForFactors.keys())[0]
            dupl1[key] = self.valuesForFactors[key][0]
            dupl2[key] = self.valuesForFactors[key][1]
            return [dupl1, dupl2]

        while len(self.unCovSets) + len(self.unCovTransitions) > 0 and len(self.coveringArray) <= self.limit:
            if self.verbose:
                coverage = 100-(len(self.unCovSets) + len(self.unCovTransitions))*100/self.totalNumberOfPairs
                print("(" + str(self.numTests) + ") Current coverage: " + str(coverage) + "%")

            #self.weight_lookahead = self.base_weight_lookahead
            #if self.prevWasDecidedByLookahead:
            #    if self.verbose:
            #        print("stop gap measure")
            #    self.weight_lookahead = 0
            #    self.prevWasDecidedByLookahead = False

            prevTestCase = None if len(self.coveringArray) == 0 else self.coveringArray[-1]
            if not self.bestOfBoth:
                testCasePool = self.generateCandidates(prevTestCase, self.numCandidates)
            else:
                print("activating bestofboth ?")
                testCasePool = self.generateCandidates(prevTestCase, round(self.numCandidates/2))
                self.weight_lookahead = 0
                testCasePool = testCasePool + self.generateCandidates(prevTestCase, round(self.numCandidates/2))
                self.weight_lookahead = self.base_weight_lookahead


            # Selects the best test case among all candidates.
            if len(testCasePool) > 0:
                bestTestCase = self.selectBestTestCase(testCasePool, prevTestCase)
                #if prevTestCase is not None:
                #    cost = sum([1 for v in bestTestCase.values() if v not in prevTestCase.values()])
                #    if cost == 0:

                improvedCoverage = self.updateUnCovSets(bestTestCase, prevTestCase)
                #if not improvedCoverage:
                #    self.weight_lookahead = 0.5
                #    #self.halfLookahead = True
                #    if self.verbose:
                #        print("activating stop gap measure to prevent immobilisation")
                #    testCasePool = self.generateCandidates(prevTestCase)
                #    improvedCoverage = self.updateUnCovSets(bestTestCase, prevTestCase)
                #    if not improvedCoverage:
                #       print("stop gab measure failed, coverage did not increase")
                #    bestTestCase = self.selectBestTestCase(testCasePool, prevTestCase)

                self.coveringArray.append(bestTestCase)
                self.numTests += 1

        return self.coveringArray.copy()

    def generateCandidates(self, prevTestCase, candidates):
        # Add first pair of factor-value to the test case.
        testCasePool = []
        for count in range(candidates):
            newTestCase = {}
            pairsScores = self.computeScores(prevTestCase)
            bestScore = max(pairsScores.values())
            bestFactor, bestValue = random.choice([key for key, value in pairsScores.items() if value == bestScore])
            newTestCase[bestFactor] = bestValue

            # Propagate this value using the SATsolver to find associated values.
            newTestCase = self.propagateCurrentTestCase(newTestCase)

            # Find a value for all other context and feature, in a random order.
            shuffledRemainingFactors = [f for f in self.factors if f not in newTestCase]
            random.shuffle(shuffledRemainingFactors)
            for f in shuffledRemainingFactors:
                v = self.selectSpecificBestValue(f, newTestCase, prevTestCase)
                newTestCase[f] = v
                if not self.solver.checkSAT(newTestCase.values()):
                    newTestCase[f] = -v

                # Propagate new values.
                newTestCase = self.propagateCurrentTestCase(newTestCase)

            # Once the test case is built, this orders the keys in the dictionary; useless but pretty to the eyes when printed.
            orderedNewTestCase = dict.fromkeys(self.systemData.getValuesForFactors())
            # Adds the core values for a complete test case.
            for key in orderedNewTestCase:
                if key in self.core:
                    orderedNewTestCase[key] = self.core[key]
                else:
                    orderedNewTestCase[key] = newTestCase[key]
            testCasePool.append(orderedNewTestCase)
        return testCasePool
