import DecisionTree as DT
import sys

if len(sys.argv) < 2:
    print("Usage python simple_testing.py <filename>")
    exit()

filename = sys.argv[1]

mytree = DT.Decision_Tree()

mytree.build(filename)
print(mytree)
print("Tree size:", mytree.size())


