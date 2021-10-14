\* The_Asset_Allocation_Model *\
Minimize
Objective_function: 2.5 nnp_0_0 + 2.5 nnp_0_1 + 2.5 nnp_0_2 + 2.5 nnp_0_3
 + 0.5 u_0 - 0.125 x_0_0_att - 0.125 x_0_0_gmc - 0.125 x_0_0_usx
 - 0.125 x_0_1_att - 0.125 x_0_1_gmc - 0.125 x_0_1_usx - 0.125 x_0_2_att
 - 0.125 x_0_2_gmc - 0.125 x_0_2_usx - 0.125 x_0_3_att - 0.125 x_0_3_gmc
 - 0.125 x_0_3_usx - x_0_att - x_0_gmc - x_0_usx
Subject To
budget_in_0: x_0_att + x_0_gmc + x_0_usx = 1
budget_in_0_0: x_0_0_att + x_0_0_gmc + x_0_0_usx - 1.3 x_0_att - 1.225 x_0_gmc
 - 1.149 x_0_usx = 0
budget_in_0_1: x_0_1_att + x_0_1_gmc + x_0_1_usx - 1.103 x_0_att
 - 1.29 x_0_gmc - 1.26 x_0_usx = 0
budget_in_0_2: x_0_2_att + x_0_2_gmc + x_0_2_usx - 1.216 x_0_att
 - 1.216 x_0_gmc - 1.019 x_0_usx = 0
budget_in_0_3: x_0_3_att + x_0_3_gmc + x_0_3_usx - 0.954 x_0_att
 - 0.728 x_0_gmc - 0.922 x_0_usx = 0
nnp_constr._in_0_0: nnp_0_0 + u_0 + x_0_0_att + x_0_0_gmc + x_0_0_usx >= 0
nnp_constr._in_0_1: nnp_0_1 + u_0 + x_0_1_att + x_0_1_gmc + x_0_1_usx >= 0
nnp_constr._in_0_2: nnp_0_2 + u_0 + x_0_2_att + x_0_2_gmc + x_0_2_usx >= 0
nnp_constr._in_0_3: nnp_0_3 + u_0 + x_0_3_att + x_0_3_gmc + x_0_3_usx >= 0
Bounds
 u_0 free
End
