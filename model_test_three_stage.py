import pulp

lmbda = 0.5
alpha = 0.05
stocks = ["att", "gmc", "usx"]
scenarios = ["s1", "s2","s3","s4"]
probs = dict(zip(scenarios, [0.25, 0.25, 0.25, 0.25]))

returns = {"s1": {"att": 1.300,
                  "gmc": 1.225,
                  "usx": 1.149},
           "s2": {"att": 1.103,
                  "gmc": 1.290,
                  "usx": 1.260},
           "s3": {"att": 1.216,
                  "gmc": 1.216,
                  "usx": 1.019},
           "s4": {"att": 0.954,
                  "gmc": 0.728,
                  "usx": 0.922},
}

"""
     att     gmc     usx
 s1  1.300   1.225   1.149
 s2  1.103   1.290   1.260
 s3  1.216   1.216   1.019
 s4  0.954   0.728   0.922
"""

first_stage_vars = pulp.LpVariable.dicts(
    "x_1", stocks, lowBound=0, cat="Continuous"
)

u_1 = pulp.LpVariable("u_1", cat="Continuous")

nnp_1 = pulp.LpVariable.dicts(
    "nnp_1", scenarios, lowBound=0, cat="Continuous"
)

second_stage_vars = pulp.LpVariable.dicts(
    "x_2", (scenarios, stocks), lowBound=0, cat="Continuous"
)

third_stage_vars = pulp.LpVariable.dicts(
    "x_3", (scenarios, scenarios, stocks), lowBound=0, cat="Continuous"
)



Q_3 = [ - pulp.lpSum(third_stage_vars[j])  for j in scenarios]
Q_3 = dict(zip(scenarios, Q_3))

R_3 = [pulp.lpSum(
        [
            probs[j] * ( (1 - lmbda) * Q_3[j][i] + (lmbda / alpha) * nnp_1[j][i])
            for i in scenarios
        ]) for j in scenarios]

Q_2 = [ - pulp.lpSum(second_stage_vars[j]) + pulp.lpSum([lmbda * u_2[j], R_3[j]]) for j in scenarios]
Q_2 = dict(zip(scenarios, Q_2))


R_2 = pulp.lpSum(
        [
            probs[j] * ( (1 - lmbda) * Q_2[j] + (lmbda / alpha) * nnp_1[j])
            for j in scenarios
        ])

Q_1 = - pulp.lpSum(first_stage_vars) + pulp.lpSum([lmbda * u_1, R_2])



my_problem = pulp.LpProblem("The_Asset_Allocation_Promlem", pulp.LpMinimize)

my_problem += (
    Q_1,
    "Objective function",
)

my_problem += (
    pulp.lpSum(first_stage_vars) == 1,
    "Q_1 budget",
)

for j in scenarios:
    my_problem += (
        pulp.lpSum(second_stage_vars[j]) == pulp.lpSum([returns[j][i] * first_stage_vars[i] for i in stocks]) ,
        "Q_2 budget in " + j,
    )

    my_problem += (
        nnp_1[j] >= Q_2[j] - u_1,
        "non-negative part constraint in " + j,
    )

print(my_problem)


my_problem.writeLP("my_problem.lp")
# The problem is solved using PuLP's choice of Solver
my_problem.solve()


for v in my_problem.variables():
    print(v.name, "=", v.varValue)

