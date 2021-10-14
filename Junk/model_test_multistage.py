import pandas as pd
import numpy as np
from anytree import Node, RenderTree, PostOrderIter
import pulp

stocks = ["att", "gmc", "usx"]
#returns = [pd.DataFrame(np.random.rand(2,3),columns = stocks) for i in range(3)]

test_returns = np.array([[1.300, 1.225, 1.149],
 [1.103, 1.290, 1.260],
 [1.216, 1.216, 1.019],
 [0.954, 0.728, 0.922]])

returns = [pd.DataFrame(test_returns, columns = stocks)]

structure = [4]
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
print(RenderTree(root))

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

#add objective function first
model += (
            root.Q,
            "Objective function",
        )

#add constraintns to the model
for node in PostOrderIter(root):
    id = "_".join([str(x) for x in node.name])

    if node.is_root:
        #budget in root
        model += (
            pulp.lpSum(node.x) == 1,
            "budget in " + id,
        )
    else:
        # budget elsewhere
        model += (
            pulp.lpSum(node.x) == pulp.lpSum(
                [returns[len(node.name) - 2].loc[node.name[-1], i] * node.parent.x[i] for i in stocks]),
            "budget in " + id,
        )

        # nnp constraint
        model += (
            node.nnp >= node.Q - node.parent.u,
            "nnp constr. in " + id,
        )



print(model)


model.writeLP("model_test.lp")

model.solve()

for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, str(node.name) + ": " + ", ".join([ " ".join([key + ":", str(node.x[key].varValue)]) for key in node.x.keys()])))
