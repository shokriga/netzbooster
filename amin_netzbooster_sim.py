import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints, write_objective, define_variables
import pandas as pd

import logging
logger = logging.getLogger(__name__)

# for testing
if 'snakemake' not in globals():
    from vresutils import Dict
    import yaml
    snakemake = Dict()
    snakemake.input = ["mnt/amin/sim-netzbooster-model/networks/prenetwork1.nc"]
    with open('mnt/amin/sim-netzbooster-model/config.yaml', encoding='utf8') as f:
        snakemake.config = yaml.safe_load(f)



def add_to_objective(n, snapshots):

    logger.info("define objective")

    n.determine_network_topology()
    sub = n.sub_networks.at["0", "obj"]
    sub.calculate_PTDF()
    sub.calculate_BODF()

    buses = sub.buses_i()
    lines = sub.lines_i()
    outages = [l for l in lines if "outage" in l]

    coefficient1 = 1
    coefficient2 = 0.0001

    terms = linexpr(
                    (coefficient1, get_var(n, "Bus", "P_pos").loc[buses]),
                    (coefficient1, get_var(n, "Bus", "P_neg").loc[buses]),
                    (coefficient2, get_var(n, "Bus", "p_pos").loc[buses].sum().sum())
                    )

    write_objective(n, terms)



#     objective.variables.extend([(coefficient1, m.P_pos[i]) for i in buses])
#     objective.variables.extend([(coefficient1, m.P_neg[i]) for i in buses])
#     objective.variables.extend([(coefficient2, m.p_pos[i, k, t])
#                                 for i in buses for k in outages for t in snapshots])

#     l_objective(m, objective)



def netzbooster_constraints(n, snapshots):

    # Define indices & parameters
    n.determine_network_topology()
    sub = n.sub_networks.at["0", "obj"]
    sub.calculate_PTDF()
    sub.calculate_BODF()

    snapshots = snapshots #n.snapshots
    buses = sub.buses_i()
    lines = sub.lines_i()
    outages = [l for l in lines if "outage" in l]

    ptdf = pd.DataFrame(sub.PTDF, index=lines, columns=buses)
    lodf = pd.DataFrame(sub.BODF, index=lines, columns=lines)

    logger.info("define variables")
    # ......
    # something like
    # define_variables(n, lower_limit, upper_limit, component (e.g. "Bus"), attr (e.g. "p"))

    logger.info("define constraints")
    # in general you define a constraint as
    lhs = linexpr(
                    (1,  get_var(n, "Bus", "p_pos").loc[(buses, outages, snapshots)]),
                    (-1, get_var(n, "Bus", "P_pos").loc[buses])
                   )
    define_constraints(n, lhs, "<=", 0., "Bus", "UpLimitPos")

    # # Define Variables
    # # P_pos(i) >= 0
    # P_pos = Var(buses, within=NonNegativeReals)

    # # P_neg(i) >= 0
    # model.P_neg = Var(buses, within=NonNegativeReals)

    # # p_pos(i,l,t) >= 0
    # model.p_pos = Var(buses, outages, snapshots, within=NonNegativeReals)

    # # p_neg(i,l,t) >= 0
    # model.p_neg = Var(buses, outages, snapshots, within=NonNegativeReals)


    # logger.info("define constraints")

    # # Define constraints
    # ########### p_pos(i,k,t) <= P_pos(i)  ##########
    # c1 = {(i, k, t):
    #       [[(1, model.p_pos[i, k, t]), (-1, model.P_pos[i])], "<=", 0.]
    #       for i in buses for k in outages for t in snapshots}

    # l_constraint(model, "UpLimitPos", c1, buses, outages, snapshots)

    # ######### p_neg(i,k,t) <= P_neg(i)  #########
    # c2 = {(i, k, t):
    #       [[(1, model.p_neg[i, k, t]), (-1, model.P_neg[i])], "<=", 0.]
    #       for i in buses for k in outages for t in snapshots}

    # l_constraint(model, "UpLimitNeg", c2, buses, outages, snapshots)

    # ######## sum(i, p_pos(i,k,t) - p_neg(i,k,t)) = 0  #########
    # c3 = {(k, t):
    #       [[*[(1, model.p_pos[i, k, t]) for i in buses],
    #         *[(-1, model.p_neg[i, k, t]) for i in buses]], "==", 0.]
    #       for k in outages for t in snapshots}
    # l_constraint(model, "FlexBal", c3, outages, snapshots)


    # ######## sum(i, PTDF(l,i) * (p_neg(i,k) - p_pos(i,k)) <= F(l) + f(l) + LODF(l,k) * f(k) ########
    # c4 = {(l, k, t):
    #       [[*[(-ptdf.at[l, i], model.p_pos[i, k, t]) for i in buses],
    #         *[(ptdf.at[l, i], model.p_neg[i, k, t]) for i in buses],
    #         [(-1, n.lines_t.p0.at[t, l])],
    #         [(-lodf.at[l, k], n.lines_t.p0.at[t, l])]
    #           ], "<=", n.lines.at[l, "s_nom"]]
    #       for l in lines for k in outages for t in snapshots}

    # l_constraint(model, "LineDn", c4, lines, outages, snapshots)

    # # sum(i, PTDF(l,i) * (p_pos(i,k) - p_neg(i,k)) <= F(l) - f(l) - LODF(l,k) * f(k)
    # c5 = {(l, k, t): [[
    #     *[(ptdf.at[l, i], m.p_pos[i, k, t]) for i in buses],
    #     *[(-ptdf.at[l, i], m.p_neg[i, k, t]) for i in buses],
    #     *[(1, n.lines_t.p0.at[t, l])],
    #     [(lodf.at[l, k], n.lines_t.p0.at[t, l])]
    #       ], "<=", n.lines.at[l, "s_nom"]]
    #       for l in lines for k in outages for t in snapshots}

    # l_constraint(model, "LineUp", c5, lines, outages, snapshots)



def extra_functionality(n,snapshots):

    netzbooster_constraints(n, snapshots)
    add_to_objective(n, snapshots)




# import network
n = pypsa.Network(snakemake.input[0])


n.lines.s_max_pu = 1

solver_options = snakemake.config["solver"]
solver_name = solver_options.pop("name")

# for testing only consider 3 snapshots
snapshots = n.snapshots[:3]

n.lopf(snapshots=snapshots,
       pyomo=False,
       solver_name=solver_name,
       solver_options=solver_options,
       extra_functionality=extra_functionality,
       formulation="kirchhoff",
       skip_objective=False,   # False assumes same objective function as before extended by your expression
      )


n.export_to_netcdf(snakemake.output[0])