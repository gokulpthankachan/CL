import pandas as pd
import numpy as np
import math

class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""
def entropy(data):
    yes=0.0
    no=0.0
    for _,rows in data.iterrows():
        if rows["Decision"]=="yes":
            yes+=1
        elif rows["Decision"]=="no":
            no+=1
    if yes==0.0 or no==0.0:
        return 0
    else:
        py=yes/(yes+no)
        pn=no/(yes+no)
        return -(py*math.log(py,2)+pn*math.log(pn,2))
def info_gain(dataset,feature):
    attributes=np.unique(dataset[feature])
    gain=entropy(dataset)
    for attr in attributes:
        subdata=dataset[dataset[feature]==attr]
        sub_e=entropy(subdata)
        gain -= (float(len(subdata)) / float(len(dataset))) * sub_e
    return gain
def ID3(dataset,features):
    root=Node()
    max_gain=0
    max_feature=""
    for feature in features:
        gain=info_gain(dataset,feature)
        if gain>max_gain:
            max_gain=gain
            max_feature=feature
    root.value=max_feature
    at= np.unique(dataset[max_feature])
    for a in at:
        subdata = dataset[dataset[max_feature] == a]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = a
            newNode.pred = np.unique(subdata["Decision"])
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = a
            new_attrs = features.copy()
            new_attrs.remove(max_feature)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root
def printTree(root: Node, depth=0):
    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    for child in root.children:
        printTree(child, depth + 1)
# def classify(root: Node, new):
#     for child in root.children:
#         if child.value == new[root.value]:
#             if child.isLeaf:
#                 print ("Predicted Label for new example", new," is:", child.pred)
#                 exit
#             else:
#                 classify (child.children[0], new)
dataset=pd.read_csv("ID3.csv")
# print("The DATASET")
# print(dataset)
# print ("------------------")
features=[feat for feat in dataset]
features.remove("Decision")
root = ID3(dataset,features)
print("\nDecision Tree is : \n")
printTree(root)
