from anytree import Node, RenderTree, PostOrderIter
import pulp

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


def create_model_and_tree(returns: [], stocks: [], parameters: dict):
    structure = [len(df.index) for df in returns]
    lmbda = parameters["lmbda"]
    alpha = parameters["alpha"]
    c = parameters["transaction_cost"]

    root = Node([0])

    # create tree structure
    create_subtree(structure, root)

    # add variables to the tree
    for node in PostOrderIter(root):
        id = "_".join([str(x) for x in node.name])
        node.x = pulp.LpVariable.dicts(
            "x_" + id, stocks, lowBound=0, cat="Continuous"
        )
        if not node.is_leaf:
            node.u = pulp.LpVariable("u_" + id, cat="Continuous")
        if not node.is_root:
            node.nnp = pulp.LpVariable("nnp_" + id, lowBound=0, cat="Continuous")
            node.o = pulp.LpVariable.dicts(
                "o_" + id, stocks, cat="Continuous"
            )

    # add equations to the tree
    for node in PostOrderIter(root):
        if node.is_leaf:
            node.Q = - pulp.lpSum(node.x)
        else:
            node.R = (1 / structure[len(node.name) - 1]) * pulp.lpSum(
                [
                    ((1 - lmbda) * child.Q + (lmbda / alpha) * child.nnp)
                    for child in node.children
                ])
            node.Q = - pulp.lpSum(node.x) + lmbda * node.u + node.R

    """"
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
    """

    #initialize the model
    model = pulp.LpProblem("The_Asset_Allocation_Model", pulp.LpMinimize)

    # add objective function first
    model += (
        root.Q,
        "Objective function",
    )

    # add constraints to the model
    for node in PostOrderIter(root):
        id = "_".join([str(x) for x in node.name])

        if node.is_root:
            # budget in root
            model += (
                pulp.lpSum(node.x) == 1,
                "budget in " + id,
            )
        else:
            # budget elsewhere
            model += (
                pulp.lpSum(node.x)  + c * pulp.lpSum(node.o) == pulp.lpSum(
                    [returns[len(node.name) - 2].loc[node.name[-1], i] * node.parent.x[i] for i in stocks]),
                "budget in " + id,
            )

            for i in stocks:
                model += (
                    node.o[i] - node.x[i] >= - node.parent.x[i],
                    "A abs stock" + str(i) + " in " + id,
                )
                model += (
                    node.o[i] + node.x[i] >= node.parent.x[i],
                    "B abs stock" + str(i) + " in " + id,
                )


            # nnp constraint
            model += (
                node.nnp >= node.Q - node.parent.u,
                "nnp constr. in " + id,
            )

    return model, root



class Model():
    def __init__(self, returns: [], stocks: [], parameters: dict):
        self.returns = returns
        self.stocks = stocks
        self.parameters = parameters

        self.model, self.root = create_model_and_tree(returns, stocks, parameters)

    def print(self):
        print(self.model)

    def print_scenarios(self):
        for scen in self.returns:
            print(scen)
            print("------")

    def save_lp_file(self, name):
        self.model.writeLP(name + ".lp")

    def solve(self):
        self.model.solve()

    def print_results_in_tree(self):
        for pre, fill, node in RenderTree(self.root):
            values = ", ".join([" ".join([key + ":", str("{:.3f}".format(node.x[key].varValue))])
                                for key in node.x.keys()])
            print("%s%s" % (pre, str(node.name) + ": " + values))
