import time
from CITSAT import CITSAT
from ResultRefining import printCoveringArray, numberOfChangements, orderArray
from SystemData import SystemData

time1 = time.time()
# models = "./data/enlarged/"
# s = SystemData(models+'contexts.txt', models+'features.txt', models+'mapping.txt')
s = SystemData('contexts.txt', 'features.txt', 'mapping.txt')
result = CITSAT(s, False, 30)
totalTime = time.time() - time1
# printCoveringArray(result, s, "Normal", order=False)
# print("================================ORDER = False=====================================")
# printCoveringArray(result, s, "Refined", writeMode=False, order=False)
# print("================================ORDER = True=====================================")
print("==================== To create a Latex table ====================")
printCoveringArray(result, s, mode="Refined", latex=True)
print("\n")
print("==================== Normal version ====================")
printCoveringArray(result, s, mode="Refined", latex=False)

print("Computation time : " + str(totalTime) + " seconds")
unrefinedCost = numberOfChangements(result, s.getContexts())
# print("COST UNREFINED : " + str(unrefinedCost))
refinedCost = numberOfChangements(orderArray(result, s.getContexts()), s.getContexts())
# print("COST REFINED : " + str(refinedCost))
# print("Decrease in cost of : " + str((unrefinedCost - refinedCost)/unrefinedCost))
