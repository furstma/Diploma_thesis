import pandas as pd
import numpy as np
from anytree import Node, RenderTree, PostOrderIter
import pulp

stocks = ["att", "gmc", "usx"]
returns = [pd.DataFrame(np.random.rand(2,3),columns = stocks) for i in range(2)]

structure = [2,2,2]
lmbda = 0.5
alpha = 0.05


root = Node([0])


def create_subtree(sub_tree_structure, parent):
    if len(sub_tree_structure) == 1:
        children = []
        for s in range(sub_tree_structure[0]):
            children.append(Node(parent.name + [s]))
        parent.children = children
    else:
        children = []
        for s in range(sub_tree_structure[0]):
            children.append(Node(parent.name + [s]))
        parent.children = children

        for child in parent.children:
            create_subtree(sub_tree_structure[1:], child)



create_subtree(structure, root)


#create variables
for node in PostOrderIter(root):
    id = "_".join([str(x) for x in node.name])
    node.x = pulp.LpVariable.dicts(
        "x_" + id , stocks, lowBound=0, cat="Continuous"
    )
    if not node.is_leaf:
        node.u = pulp.LpVariable("u_" + id, cat="Continuous")
    if not node.is_root:
        node.nnp = pulp.LpVariable("nnp_" + id, lowBound=0, cat="Continuous")

#add equations to the tree
for node in PostOrderIter(root):
    id = "_".join([str(x) for x in node.name])
    if node.is_leaf:
        node.Q = - pulp.lpSum(node.x)
    else:
        node.R = (1 / structure[len(node.name)-1]) * pulp.lpSum(
        [
            ( (1 - lmbda) * child.Q + (lmbda / alpha) * child.nnp)
            for child in node.children
        ])
        node.Q = - pulp.lpSum(node.x) + lmbda * node.u + node.R



for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
    print("%s%s" % (fill, node.x))
    if not node.is_leaf:
        print("%s%s" % (fill, node.u))
    if not node.is_root:
        print("%s%s" % (fill, node.nnp))
    print("%s%s" % (fill, node.Q))
    if not node.is_leaf:
        print("%s%s" % (fill, node.R))




#initialize the model
model = pulp.LpProblem("The_Asset_Allocation_Model", pulp.LpMinimize)


#add equations and constraintns to the model
for node in PostOrderIter(root):
    id = "_".join([str(x) for x in node.name])

    if node.is_leaf:
        node.Q = - pulp.lpSum(node.x)
    if not node.is_root:
        #budget
        model += (
            pulp.lpSum(node.x) == pulp.lpSum([returns[len(node.name)-1].loc[model.name[-1], i] * node.parent.x[i] for i in stocks]),
            "budget in " + id,
        )


