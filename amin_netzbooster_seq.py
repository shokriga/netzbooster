import pypsa
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Constraint,
                           Reals, Suffix, Binary, SolverFactory)
from pypsa.opt import (l_constraint, l_objective, LExpression, LConstraint)
import pandas as pd

from vresutils.benchmark import memory_logger

import logging
logger = logging.getLogger(__name__)



def prepare_model():

#     n = pypsa.Network("../../co2-analysis/results/new/postnetwork_heur125.nc")
    n = pypsa.Network(snakemake.input[0])
    
    # n = pypsa.Network("postnetwork_heur1.0.nc")
#     n = get_representative_snapshots(n, N_SNAPSHOTS)

    n.determine_network_topology()
    sub = n.sub_networks.at["0", "obj"]

    sub.calculate_PTDF()
    sub.calculate_BODF()

    # Define indices
    snapshots = n.snapshots
    buses = sub.buses_i()
    lines = sub.lines_i()
    outages = [l for l in lines if "outage" in l]

    ptdf = pd.DataFrame(sub.PTDF, index=lines, columns=buses)
    lodf = pd.DataFrame(sub.BODF, index=lines, columns=lines)

    logger.info("define variables")

    # define model
    m = ConcreteModel()

    # Define Variables
    # P_pos(i) >= 0
    m.P_pos = Var(buses, within=NonNegativeReals)

    # P_neg(i) >= 0
    m.P_neg = Var(buses, within=NonNegativeReals)

    # p_pos(i,l,t) >= 0
    m.p_pos = Var(buses, outages, snapshots, within=NonNegativeReals)

    # p_neg(i,l,t) >= 0
    m.p_neg = Var(buses, outages, snapshots, within=NonNegativeReals)

    logger.info("define constraints")

    # Define constraints
    ########### p_pos(i,k,t) <= P_pos(i)  ##########
    c1 = {(i, k, t):
          [[(1, m.p_pos[i, k, t]), (-1, m.P_pos[i])], "<=", 0.]
          for i in buses for k in outages for t in snapshots}

    l_constraint(m, "UpLimitPos", c1, buses, outages, snapshots)

    ######### p_neg(i,k,t) <= P_neg(i)  #########
    c2 = {(i, k, t):
          [[(1, m.p_neg[i, k, t]), (-1, m.P_neg[i])], "<=", 0.]
          for i in buses for k in outages for t in snapshots}

    l_constraint(m, "UpLimitNeg", c2, buses, outages, snapshots)

    ######## sum(i, p_pos(i,k,t) - p_neg(i,k,t)) = 0  #########
    c3 = {(k, t):
          [[*[(1, m.p_pos[i, k, t]) for i in buses],
            *[(-1, m.p_neg[i, k, t]) for i in buses]], "==", 0.]
          for k in outages for t in snapshots}
    l_constraint(m, "FlexBal", c3, outages, snapshots)

    ######## sum(i, PTDF(l,i) * (p_neg(i,k) - p_pos(i,k)) <= F(l) + f(l) + LODF(l,k) * f(k) ########
    c4 = {(l, k, t):
          [[*[(-ptdf.at[l, i], m.p_pos[i, k, t]) for i in buses],
            *[(ptdf.at[l, i], m.p_neg[i, k, t]) for i in buses
              ]], "<=", n.lines_t.p0.at[t, l] + lodf.at[l, k] * n.lines_t.p0.at[t, k] + n.lines.at[l, "s_nom"]]
          for l in lines for k in outages for t in snapshots}

    l_constraint(m, "LineDn", c4, lines, outages, snapshots)

    # sum(i, PTDF(l,i) * (p_pos(i,k) - p_neg(i,k)) <= F(l) - f(l) - LODF(l,k) * f(k)
    c5 = {(l, k, t): [[
        *[(ptdf.at[l, i], m.p_pos[i, k, t]) for i in buses],
        *[(-ptdf.at[l, i], m.p_neg[i, k, t]) for i in buses
          ]], "<=", -n.lines_t.p0.at[t, l] - lodf.at[l, k] * n.lines_t.p0.at[t, k] + n.lines.at[l, "s_nom"]]
          for l in lines for k in outages for t in snapshots}

    l_constraint(m, "LineUp", c5, lines, outages, snapshots)

    logger.info("define objective")

    objective = LExpression()

    coefficient1 = 1
    coefficient2 = 0.0001

    objective.variables.extend([(coefficient1, m.P_pos[i]) for i in buses])
    objective.variables.extend([(coefficient1, m.P_neg[i]) for i in buses])
    objective.variables.extend([(coefficient2, m.p_pos[i, k, t])
                                for i in buses for k in outages for t in snapshots])

    l_objective(m, objective)

    return m


def get_values(indexedvar):
    return pd.Series(indexedvar.get_values())


def pyomo_postprocess(options=None, instance=None, results=None):
    m.P_pos.display()
    m.P_neg.display()

    p = get_values(m.p_pos) - get_values(m.p_neg)
    p.index.set_names(["bus", "outage", "snapshot"], inplace=True)
    p = p.unstack("bus")
#     p.to_csv("new/p.csv")

    P = pd.concat({"positive": get_values(m.P_pos),
                   "negative": get_values(m.P_neg)}, axis=1)
#     P.to_csv("new/P.csv")
    P.to_csv(snakemake.output[0])


if __name__ == '__main__':

    suffix = "v07"

    memory_log_filename = f"memory-{suffix}.log"
    gurobi_log_filename = f"gurobi-{suffix}.log"

    with memory_logger(filename=memory_log_filename, interval=1.) as mem:

        m = prepare_model()

        # This emulates what the pyomo command-line tools does
        options = {"threads": 4, "method": 2,
                   "crossover": 0, "BarConvTol": 1.e-3,
                   "FeasibilityTol": 1.e-5, "AggFill": 0,
                   "PreDual": 0, "GURO_PAR_BARDENSETHRESH": 200}
        opt = SolverFactory('gurobi', options=options)

        logger.info("start solving")
        results = opt.solve(m, logfile=gurobi_log_filename, options=options)
        logger.info("solving completed")

        # sends results to stdout
        results.write()
        print("\nDisplaying Solution\n" + '-'*60)

        pyomo_postprocess(options, m, results)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
