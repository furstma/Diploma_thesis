\* The_Asset_Allocation_Promlem *\
Minimize
Objective_function: 2.5 nnp_1_s1 + 2.5 nnp_1_s2 + 2.5 nnp_1_s3 + 2.5 nnp_1_s4
 + 0.5 u_1 - x_1_att - x_1_gmc - x_1_usx - 0.125 x_2_s1_att - 0.125 x_2_s1_gmc
 - 0.125 x_2_s1_usx - 0.125 x_2_s2_att - 0.125 x_2_s2_gmc - 0.125 x_2_s2_usx
 - 0.125 x_2_s3_att - 0.125 x_2_s3_gmc - 0.125 x_2_s3_usx - 0.125 x_2_s4_att
 - 0.125 x_2_s4_gmc - 0.125 x_2_s4_usx
Subject To
Q_1_budget: x_1_att + x_1_gmc + x_1_usx = 1
Q_2_budget_in_s1: - 1.3 x_1_att - 1.225 x_1_gmc - 1.149 x_1_usx + x_2_s1_att
 + x_2_s1_gmc + x_2_s1_usx = 0
Q_2_budget_in_s2: - 1.103 x_1_att - 1.29 x_1_gmc - 1.26 x_1_usx + x_2_s2_att
 + x_2_s2_gmc + x_2_s2_usx = 0
Q_2_budget_in_s3: - 1.216 x_1_att - 1.216 x_1_gmc - 1.019 x_1_usx + x_2_s3_att
 + x_2_s3_gmc + x_2_s3_usx = 0
Q_2_budget_in_s4: - 0.954 x_1_att - 0.728 x_1_gmc - 0.922 x_1_usx + x_2_s4_att
 + x_2_s4_gmc + x_2_s4_usx = 0
non_negative_part_constraint_in_s1: nnp_1_s1 + u_1 + x_2_s1_att + x_2_s1_gmc
 + x_2_s1_usx >= 0
non_negative_part_constraint_in_s2: nnp_1_s2 + u_1 + x_2_s2_att + x_2_s2_gmc
 + x_2_s2_usx >= 0
non_negative_part_constraint_in_s3: nnp_1_s3 + u_1 + x_2_s3_att + x_2_s3_gmc
 + x_2_s3_usx >= 0
non_negative_part_constraint_in_s4: nnp_1_s4 + u_1 + x_2_s4_att + x_2_s4_gmc
 + x_2_s4_usx >= 0
Bounds
 u_1 free
End
