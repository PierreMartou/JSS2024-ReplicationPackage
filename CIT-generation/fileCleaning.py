import os

def cleanModels():
    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    bl = ["model_20161127_1823059506.txt", "model_20130703_1107240012.txt"]

    for filename in bl:  # os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)
        os.remove(txt)
        os.remove(txtConstraints)

def cleanSuites(iter=None, mode=None):
    if iter is None:
        iterations = range(3)
    else:
        iterations = [iter]
    if mode is None:
        modes = ["0", "1", "1&2", "1&2&3"]
    else:
        modes = mode

    modelFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txt/"
    constraintsFiles = "../data/SPLOT/SPLOT-NEW/SPLOT-txtconstraints/"
    storageCIT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCIT/"
    storageCTT = "../data/SPLOT/SPLOT-NEW/SPLOT-TestSuitesCTT/"
    for filename in os.listdir(modelFiles):
        txt = os.path.join(modelFiles, filename)
        txtConstraints = os.path.join(constraintsFiles, filename)

        for iteration in iterations:
            tempStorageCIT = storageCIT + filename[:-4] + "-"
            filepath = tempStorageCIT + str(iteration) + ".pkl"
            if os.path.exists(filepath):
                os.remove(filepath)


            for mode in modes:
                filepath = storageCTT + filename[:-4] + "-" + mode + "-" + str(iteration) + ".pkl"
                if os.path.exists(filepath):
                    os.remove(filepath)


if __name__ == '__main__':
    cleanSuites(mode=["1&2&3"])

    #cleanSuites(iter=2)

