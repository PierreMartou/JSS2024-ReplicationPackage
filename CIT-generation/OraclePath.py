class OraclePath:
    def __init__(self, path):
        self.path = path
        self.shortenedPath = self.createShortenedPath()

    def getInitState(self):
        return self.path[0].copy()

    def getFinalState(self):
        return self.path[-1].copy()

    def getStates(self):
        return self.path.copy()

    def getState(self, i):
        if i<0 or i>=len(self.path):
            print("The path is not that long.")
            return 0
        return self.path[i].copy()

    def createShortenedPath(self):
        newPath = [self.path[0]]
        prevP = self.path[0]
        for p in self.path:
            equivalent = True
            for s in p:
                if p[s] != prevP[s]:
                    equivalent = False
            if not equivalent:
                newPath.append(p)
        return newPath

    def getLength(self):
        return len(self.shortenedPath)
