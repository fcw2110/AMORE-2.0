#import necessary packages
import numpy as np
import sys
import math
import re
import isoprene_rates as rate
from math import exp as EXP
from copy import deepcopy
import sympy as sym
import networkx as nx
import matplotlib.pyplot as plt
#import graphviz 
#import pygraphviz as pgv
#import to_precision
import time
import csv
import pandas as pd
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import csgraph_from_dense
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
#from sklearn.preprocessing import normalize

def AMORE_mechanism_reduction(mechanism, background_spc_n, conditions, settings):
    # Version 2.0
    species_list_names_2 = deepcopy(mechanism.species)
    for i in settings['Categories']:
        species_list_names_2.append(i[0])
    species_list_names_2.append('TETRA')
    species_list_names_2.append('ISOPN')
    species_list_names_2.append('IHN')
    red_mech_size = settings['Mechanism Size']
    default_settings = {}
    default_settings['No Group'] = []
    default_settings['Manual Groups'] = []
    default_settings['No Counts'] = set()
    default_settings['Remove Species'] = []
    default_settings['Remove Reactions'] = True
    default_settings['Remove Weak Reactions'] = False
    default_settings['Weak Reaction Cutoff'] = 0
    default_settings['Keep Cycle Reactions'] = True
    default_settings['Print Progress'] = True
    default_settings['Reduce Stiffness'] = False
    default_settings['Stiffness Threshold'] = 5
    default_settings['Iterations'] = 0
    for i in default_settings:
        if i not in settings:
            settings[i] = default_settings[i]
    
    if settings['Print Progress'] == True:
        print('Stage 1: Mechanism Preprocessing')
    
    needed_concentrations = set() # any species which is the second reactant in a reaction but the concentration is not provided
    no_counts = settings['No Counts']
    # We are duplicating the mechanism and calling it test mechanism
    # This is to avoid inadvertently modifying the original mechanism variable
    test_reactions = []
    not_rxns = set(settings['Background Rxns'])
    for i in settings['Aerosol Rxns']:
        not_rxns.add(i)
    for i in range(len(mechanism.reactions)):
        if i not in not_rxns:
            test_reactions.append(deepcopy(mechanism.reactions[i]))      
    test_mechanism = Mechanism(deepcopy(mechanism.species),test_reactions)

    # number of reactions, number of species, species name to index dictionary, species list as indices
    reac_len = len(test_mechanism.reactions)
    spec_len = len(test_mechanism.species)
    dic = {test_mechanism.species[i]:i for i in range(spec_len)}
    species_list = [i for i in range(spec_len)]
    background_spc = [test_mechanism.species.index(b) for b in background_spc_n]
    # species that cannot be removed from the mechanism (manual input)
    protected = set(settings['Protected']).union(background_spc_n)
    protected = [dic[i] for i in protected]

    # species that cannot be placed in a group
    no_group = [dic[i] for i in settings['No Group']]

    # species that have been manually added to a group
    manual_groups = []
    man_group_set = set()
    for i in settings['Manual Groups']:
        g = []
        for j in i:
            g.append(dic[j])
            man_group_set.add(dic[j])
        manual_groups.append(g)
    # root species of the mechanism
    roots = [dic[i] for i in settings['roots']]

    # number of conditions
    l_c = len(conditions)

    # set of background species
    back_set = set()
    for i in background_spc_n:
        back_set.add(mechanism.species.index(i))

    # list of reactants, list of products, list of product coefficients, and product dictionary for each reaction
    # eg: rxn no. 7 is A + X --> 0.5 B + 0.2 C and X is a background species
    # eg cont: A has index 0, B has index 1, c has index 2, and X has index 3
    # eg cont:  reac_list_n[7] = [A,X]     prod_list_n[7] = [B,C]      prod_coeff_list[7] = [0.5, 0.2]     prod_dict[7] = {B:0.5, C:0.2}
    # eg cont:    reac_list[7] = [0,3]       prod_list[7] = [1, 2]     
    # eg cont: reac_no_back[7] = [0]      prod_no_back[7] = [1,2]
    # eg cont: rxn_reac[0] = {7...}, rxn_reac[3] = {7...}, rxn_prod[1] = {7...}, rxn_prod[2] = {7...}
    reac_list_n = []
    prod_list_n = []
    prod_coeff_list = []
    prod_dict = [{} for i in range(reac_len)]
    for i in range(reac_len):
        test_mechanism.reactions[i].rate = []
        reac_list_n.append(test_mechanism.reactions[i].reactants)
        prod_list_n.append(list(test_mechanism.reactions[i].prod_dict))
        for k in test_mechanism.reactions[i].prod_dict:
            prod_dict[i][dic[k]] = test_mechanism.reactions[i].prod_dict[k]
    reac_list, prod_list, reac_no_back, prod_no_back, rxn_reac, rxn_prod = rxn_index_convert(reac_list_n,prod_list_n,background_spc_n, background_spc, reac_len, spec_len, dic)
    rxn_prod = [set(i) for i in rxn_prod]
    rxn_reac = [set(i) for i in rxn_reac]

    # here we measure the relative rate of all reactions under each condition(evaluating rate constant and multiplying by secondary reactant concentration) 
    c_count = 0
    for i in conditions: 
        M = pressure_to_m(i['pressure'],i['temp'])
        TEMP = i['temp']
        SUN = i['sun']
        CFACTOR = 2.5e+19
        p_fac = M/1000000000
        double_rxns = []
        count = 0
        for j in test_mechanism.reactions:
            # if j.rate_law is null, use the pre-evaluated rate constants in eval_rate_law. Otherwise, the algorithm will evaluate the rate constants
            if j.rate_law!='null':
                if 'J' in j.rate_law:
                    j.eval_rate_law = solve_j_rate(j.rate_law, i['sza'],i['sun'])
                else:
                    j.eval_rate_law = eval(j.rate_law)
                if len(j.reactants)==1:
                    j.rate.append(j.eval_rate_law)

                elif j.reactants[1] in background_spc_n and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[1]]*p_fac)

                elif j.reactants[0] in background_spc_n and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[0]]*p_fac)

                elif j.reactants[1] in i and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[1]]*p_fac)

                elif j.reactants[0] in i and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[0]]*p_fac)

                else:
                    needed_concentrations.add(j.reactants[1])
                    root_conc = 0
                    for r in settings['roots']:
                        root_conc = i[r] + root_conc
                    j.rate.append(j.eval_rate_law*0.005*root_conc*p_fac)
                    double_rxns.append(count)
                count = count + 1
            else:
                counted_reactants = 0
                for s in j.reactants:
                    if s not in no_counts:
                        counted_reactants+=1
                if counted_reactants==1:
                    j.rate.append(j.eval_rate_law[c_count])
                elif j.reactants[1] in background_spc_n and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in background_spc_n and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[0]]*p_fac)
                elif j.reactants[1] in i and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in i and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[0]]*p_fac)
                else:
                    needed_concentrations.add(j.reactants[1])
                    root_conc = 0
                    for r in settings['roots']:
                        root_conc = i[r] + root_conc
                    j.rate.append(j.eval_rate_law[c_count]*0.05*root_conc*p_fac)

                    double_rxns.append(count)
        c_count+=1

    #This will list any secondary reactant whose concentration is not provided. The concentration is assumed to be 0.005* the root concentration if not provided
    if len(needed_concentrations)>0:
        print('For better results, add concentrations for these species: ', needed_concentrations)
    
    if settings['Print Progress'] == True:
        print('Stage 2: Representing Mechanism as Graph')

    # graph data structures. One graph is created for each condition because the conditions change the edge weightings.
    # graphs: one graph for each condition, formatted as a list of dictionaries. The list index indicates the species which is the parent species of the edge
    # the dictionary keys represent the child species of the edge, and the weightings are the dictionary values
    # example, edge A--> B with weighting 0.5 would be input as {B:0.5} at index A in the graph list.
    # in_graph: a list with the index indicating the child species and the element being a list of the parent species of that child
    # example, edge A--> B would have [A] at index B.
    # out_graph a list with the index indicating the parent species and the element being a list of the child species of that parent
    # example, edge A--> B would have [B] at index A.
    # in/out_graph_type: identical structure to the in/out_graphs,
    #the values list the reaction type, as defined by the secondary reactant or some other identifier
    # example, edge A + OH --> B would have ['OH'] at index B (in_graph_type) or index A (out_graph_type).
    graphs = [[] for c in range(l_c)]
    in_graph = [set() for i in range(spec_len)]
    out_graph = [set() for i in range(spec_len)]
    out_graph_type = [{} for i in range(spec_len)]
    in_graph_type  = [{} for i in range(spec_len)]
    
    
    
    rxn_types = {}
    species_types = {i:set() for i in range(spec_len)}
    for i in range(spec_len):
        if i not in back_set:
            edges = [{} for c in range(l_c)]
            rate_sums = [0 for c in range(l_c)]
            
            for j in rxn_reac[i]:
                for c in range(l_c):
                    # the rate sum for species i is calculated under each condition c by summing the relative rates of reactions where i is a reactant
                    rate_sums[c] = rate_sums[c] + test_mechanism.reactions[j].rate[c]
            for j in rxn_reac[i]:
                mults = [0 for c in range(l_c)]
                for c in range(l_c):
                    mults[c] = test_mechanism.reactions[j].rate[c]/max(1e-20,rate_sums[c])
                if len(test_mechanism.reactions[j].reactants) == 1:
                    type_r = 'solo'
                else:
                    mark = True
                    for p in reac_list[j]:
                        if p in background_spc:
                            type_r = p
                            mark = False
                    if mark:
                        type_r = 'double'
                rxn_types[j] = type_r
                species_types[i].add(str(type_r))
                for k in prod_dict[j]:
                    if i in in_graph_type[k]:
                        in_graph_type[k][i].add(type_r)
                    else:
                        in_graph_type[k][i] = set([type_r])
                    for c in range(l_c):
                        if k in edges[c]:
                            if c==0:
                                # here the edge is added. The weighting of species k for edge [i, k] is the product coefficient multiplied by the
                                # relative rate of that reaction j over the sum of relative rates, rate_sum. 
                                #If k is a product in multiple reactions where i is a reactant, then the edge weighting is added to.
                                edges[c][k] = edges[c][k] + prod_dict[j][k]*mults[c] 
                                out_graph[i].add(k)
                                in_graph[k].add(i)
                                out_graph_type[i][k].add(type_r)
                            else:
                                edges[c][k] = edges[c][k] + prod_dict[j][k]*mults[c]

                        
                        else:
                            if c==0:
                                edges[c][k] = prod_dict[j][k]*mults[c]
                                out_graph[i].add(k)
                                in_graph[k].add(i)
                                out_graph_type[i][k] = set([type_r])
                            else:
                                edges[c][k] = prod_dict[j][k]*mults[c]
            for c in range(l_c):
                graphs[c].append(edges[c])
        else:
            for c in range(l_c):
                graphs[c].append({})
    for i in range(len(species_types)):
        species_types[i] = sorted(list(species_types[i]))
    avg_graph = []
    # avg_graph averages the graphs from each condition to create one representative graph. 
    #The edges are the same between each condition, it is just the edge weightings which are averaged
    for i in range(spec_len):
        avg = {}
        for c in range(l_c):
            for j in graphs[c][i]:
                if j in avg:
                    avg[j]+=graphs[c][i][j]
                else:
                    avg[j] = graphs[c][i][j]
        for x in avg:
            avg[x]*=1/l_c
        avg_graph.append(avg)

    # the matrix of the mechanism graph is generally quite sparse. 
    #lil_matrix and related formats are efficient methods of representing sparse matrices
    # this is useful for efficient graph theory algorithms
    graph_mat_lil = lil_matrix((spec_len,spec_len), dtype=None)

    edges = []
    for i in range(len(out_graph)):
        for j in out_graph[i]:
            edges.append((i,j))

    count = 0
    for i,j in edges:
        graph_mat_lil[i,j] = 1

    graph_mat = graph_mat_lil.tocsr(copy=False)

    # this identifies the strongly connected components of the mechanism graph
    # scc: a list of the strongly connected components, with each element being a list of the species within the strongly connected component
    # scc_dict_2: a dictionary saying which scc (by index) every species is in, if it is in an scc. Used to speed up searches
    # scc_set: the set of all species that are in a scc.
    # in_cycle_specs: for each scc, these are the species that are in the scc with parent species outside of the scc
    # scc_out_specs: for each scc, these are the species which are children of species in the scc, but are not in the scc
    scc_result = connected_components(graph_mat, directed=True, connection='strong', return_labels=True)

    scc_dic = {}
    for i in range(len(scc_result[1])):
        if scc_result[1][i] in scc_dic:
            scc_dic[scc_result[1][i]].append(i)
        else:
            scc_dic[scc_result[1][i]] = [i]
    
    scc = []
    for i in scc_dic:
        if len(scc_dic[i])>1:
            scc.append(scc_dic[i])

    in_cycle_specs = []
    scc_out_specs = []
    for i in scc:
        count = count + 1
        in_cycle_spec = set()
        out_spec = set()
        for p in i:
            for k in in_graph[p]:
                if k not in i:
                    in_cycle_spec.add(p)
            for k in out_graph[p]:
                if k not in i:
                    out_spec.add(k)
        in_cycle_specs.append(in_cycle_spec)
        scc_out_specs.append(out_spec)
        
    

    scc_set = set()
    scc_dict_2 = {}
    for i in range(len(scc)):
        for j in scc[i]:
            scc_dict_2[j] = i
            scc_set.add(j)

    # this identifies the shortest paths to reach each species in the mechanism graph
    distance,preds,root_specs = dijkstra(graph_mat, directed=True, indices=roots, return_predecessors=True, unweighted=True, limit=np.inf, min_only=True)    
    

    scc_lens = [len(i) for i in scc]
    
    #the following code restates the shortest path in terms of the type of reactions along that shortest path
    # species groupings are created based on those which share an identical shortest path
    # paths: for each species, the list of species traversed on the shortest path 
    # path_with_types: paths restated as a combination of edge types
    # path_type_strings: paths restated as strings to identify identical paths
    paths = []
    path_roots = []
    for i in range(spec_len):
        if distance[i]<100:
            if preds[i]<0:
                path = []
            else:
                marker = preds[i]
                path = [marker]
                counter = 0
                while marker not in roots and counter<100:
                    marker = preds[marker]
                    path.append(marker)
            paths.append(path)
            path_roots.append(root_specs[i])
        else:
            paths.append([])
            path_roots.append('none')
    
    paths_with_types = []
    for i in range(len(paths)):
        path = []
        count = 0
        for j in paths[i]:
            if count == 0:
                path.append(in_graph_type[i][j])
            else:
                path.append(in_graph_type[paths[i][count-1]][j])
            count = count + 1
        path.insert(0,path_roots[i])
        species_type_string = ''
        for j in species_types[i]:
            species_type_string+=j+' '
        
        #path.append(species_type_string)
        paths_with_types.append(path)

    path_type_strings = []
    for i in paths_with_types:
        string = str(i[0]) + '+'
        for j in i[1:]:
            for k in j:
                string = string + str(k) + ','
            string = string+'+'
        path_type_strings.append(string)

    path_sim_dict = {}
    for i in range(len(path_type_strings)):
        if i not in back_set:
            if path_type_strings[i] in path_sim_dict:
                path_sim_dict[path_type_strings[i]].append(i)
            else:
                path_sim_dict[path_type_strings[i]]= [i]
    # groups: a list of groups, where each species in the group contains the same shortest path type
    groups = []  
    for i in path_sim_dict:
        if len(path_sim_dict[i])>1:
            groups.append(path_sim_dict[i])
    
    del paths_with_types

    
    if settings['Print Progress'] == True:
        print('Stage 3: Measuring Mechanism Yields')

    county = 0
    yields = []
    yields_for_order = []
    yields_for_order_2 = []
    cycle_out_graphs = [{} for c in range(l_c)]
    cycle_dag_graphs = [{} for c in range(l_c)]
    cycle_dag_graphs_2 = [{} for c in range(l_c)]
    new_graphs = []
    new_ins = []
    new_outs = []
    for c in range(l_c):
        # yields are measured for each condition
        county +=1        
        new_graph = []
        for i in graphs[c]:
            dicy = {}
            for j in i:
                dicy[j] = i[j]
            new_graph.append(dicy)
        new_graph_2 = deepcopy(new_graph)
        new_graph_3 = deepcopy(new_graph)

        new_in = []
        for i in in_graph:
            sety = set()
            for j in i:
                sety.add(j)
            new_in.append(sety)
        

        new_out = []
        for i in out_graph:
            sety = set()
            for j in i:
                sety.add(j)
            new_out.append(sety)
        
        # This for loop is used to represent the mechanism graph as a directed acyclic graph by representing each scc as a dag subsection
        for i in range(len(scc)):
            counte = 0
            
            for x in in_cycle_specs[i]:
                counte +=1
                new_out[x] = scc_out_specs[i].union(set(scc[i]))
                leny = int(np.sqrt(len(scc[i])))
                data, in_cyc_data, out_data = cycle_simulator_3_out(scc[i], x, graphs[c],out_graph,in_graph, [leny+40,2*(leny+40)],1e-6,set(scc[i]).union(scc_out_specs[i]),scc_out_specs[i])
                # this line simulates the scc, the only modifiable inputs are the 6th and 7th inputs 
                # 6th input: how many iterations at which to measure the yield (two numbers required). 
                # Default here is the length of the scc + 40 and twice that value. Smaller numbers will have shorter runtimes
                # but if they are too low, the output will be innaccurate
                # 7th input: the fraction of total mass at which point the algorithm will stop carrying over the mass, here 1e-6. 
                # Lower values will be more accurate but will also slow. 
                cycle_out_graphs[c][x] = data
                ful_dat = {}
                for p in data:
                    ful_dat[p] = data[p]
                ful_dat_2 = deepcopy(ful_dat)
                for p in in_cyc_data:
                    ful_dat[p] = in_cyc_data[p]
                for p in out_data:
                    ful_dat_2[p] = out_data[p]
                # new_graph will assign a yield of 0 to any species in the scc that does not have a parent outside of the scc
                new_graph[x]= data
                # new_graph_2 will assign a yield to any species within the scc proportional to the amount of mass that passed through it during the simulation
                new_graph_2[x]= ful_dat
                # new_graph_3 will assign a yield to any species within the scc proportional to the amount of mass that left the cycle through it
                new_graph_3[x] = deepcopy(ful_dat_2)
                cycle_dag_graphs[c][x] = ful_dat
                cycle_dag_graphs_2[c][x] = ful_dat_2

                
            # here, non-DAG edges are removed
            for y in scc[i]:
                if y not in in_cycle_specs[i]:

                    for s in out_graph[y]:
                        if y in new_in[s]:
                            new_in[s].remove(y)
                    new_in[y] = set(in_cycle_specs[i])
                    new_out[y] = set()
                    new_graph[y] = set()
                    new_graph_2[y]= set()
        new_graphs.append(new_graph)
        new_ins.append(new_in)
        new_outs.append(new_out)
        yi = get_yields(roots, 1e-8, new_graph, new_in, new_out, back_set,scc,scc_dict_2, scc_set, spec_len)
        yi_2 = get_yields(roots, 1e-8, new_graph_2, new_in, new_out, back_set,scc,scc_dict_2, scc_set, spec_len)
        yi_3 = get_yields(roots, 1e-8, new_graph_3, new_in, new_out, back_set,scc,scc_dict_2, scc_set, spec_len)
        # the yields are calculated with three different dag-graphs based on how interior scc species are treated
        yields.append(yi)
        yields_for_order.append(yi_2)
        yields_for_order_2.append(yi_3)

    # this isolates the dag-ified edges of the scc species with parents outside of the scc
    avg_cycle_out_graph = {}
    for c in range(l_c):
        for x in cycle_out_graphs[c]:
            
            if x in avg_cycle_out_graph:
                for j in cycle_out_graphs[c][x]:
                    if j in avg_cycle_out_graph[x]:
                        avg_cycle_out_graph[x][j]+=cycle_out_graphs[c][x][j]
                    else:
                        avg_cycle_out_graph[x][j] = cycle_out_graphs[c][x][j]
            else:
                avg_cycle_out_graph[x] = {}
                for j in cycle_out_graphs[c][x]:
                    if j in avg_cycle_out_graph[x]:
                        avg_cycle_out_graph[x][j]+=cycle_out_graphs[c][x][j]
                    else:
                        avg_cycle_out_graph[x][j] = cycle_out_graphs[c][x][j]
    
    for i in avg_cycle_out_graph:
        for j in avg_cycle_out_graph[i]:
            avg_cycle_out_graph[i][j]*=1/l_c



    # this is a different representation of the above, where the edges include the total yields including mass that passed through the scc species
    avg_cycle_dag_graph = {}
    for c in range(l_c):
        for x in cycle_dag_graphs[c]:
            
            if x in avg_cycle_dag_graph:
                for j in cycle_dag_graphs[c][x]:
                    if j in avg_cycle_dag_graph[x]:
                        avg_cycle_dag_graph[x][j]+=cycle_dag_graphs[c][x][j]
                    else:
                        avg_cycle_dag_graph[x][j] = cycle_dag_graphs[c][x][j]
            else:
                avg_cycle_dag_graph[x] = {}
                for j in cycle_dag_graphs[c][x]:
                    if j in avg_cycle_dag_graph[x]:
                        avg_cycle_dag_graph[x][j]+=cycle_dag_graphs[c][x][j]
                    else:
                        avg_cycle_dag_graph[x][j] = cycle_dag_graphs[c][x][j]
    
    for i in avg_cycle_dag_graph:
        for j in avg_cycle_dag_graph[i]:
            avg_cycle_dag_graph[i][j]*=1/l_c



    
    # these are the yields used to determine the species removal order. It includes the yields associated with scc species without parents outside of the scc
    # this is used to prevent scc species from being removed prematurely
    avg_yield_o = {}
    for i in yields_for_order[0]:
        specyi = []
        for c in range(l_c):
            specyi.append(yields_for_order[c][i])
        avg_yield_o[i] = np.mean(specyi)

    # this is an alternate method of incorporating scc species into yields
    avg_yield_o2 = {}
    for i in yields_for_order_2[0]:
        specyi = []
        for c in range(l_c):
            specyi.append(yields_for_order_2[c][i])
        avg_yield_o2[i] = np.mean(specyi)

    # boost_factor: the last member of a group to be removed will be boosted in priority by this factor
    boost_factor = 1.5
    ordered_species = []
    ordered_yields = []
    yield_values = []
    for i in avg_yield_o:
        yield_values.append(avg_yield_o[i])
    max_yield = max(yield_values)
    for i in avg_yield_o:
        if i in protected:
            ordered_yields.append(max_yield)
        else:
            ordered_yields.append(avg_yield_o[i])
        ordered_species.append(i)


    # this is the average yield of each species, ignoring interior scc species whose yield is not well defined but extrapolated in 
    avg_yield = {}
    for i in yields[0]:
        specyi = []
        for c in range(l_c):
            specyi.append(yields[c][i])
        avg_yield[i] = np.mean(specyi)
        
    
    
    ordered_species = [x for _, x in sorted(zip(ordered_yields, ordered_species))]
    ordered_yields.sort()
    spec_order_dict = {}
    count = 0
    for i in ordered_species:
        spec_order_dict[i] = count
        count+=1
    for i in groups:
        min_dict = {}
        for j in i:
            min_dict[spec_order_dict[j]] = j
        min_spec = min_dict[min(min_dict)]
        ordered_yields[spec_order_dict[min_spec]] *= boost_factor

    # ordered_species: the order in which to remove species
    ordered_species = [x for _, x in sorted(zip(ordered_yields, ordered_species))]
    for i in protected:
        ordered_species.remove(i)
        ordered_species.append(i)
    ordered_yields.sort()

    for i in settings['Remove Species']:
        ordered_species.remove(dic[i])
        ordered_species.insert(0, dic[i])

    # a copy of the graph objects to be reduced
    reduced_graph, reduced_in, reduced_out = copy_graph(avg_graph, in_graph, out_graph)
    reduced_reactions = []
    for i in test_mechanism.reactions:
        reduced_reactions.append(deepcopy(i))

    # copy of mechanism to be reduced
    reduced_mech = Mechanism(deepcopy(test_mechanism.species),reduced_reactions)

    for i in range(reac_len):
        reduced_mech.reactions[i].prod_dict = prod_dict[i]
        reduced_mech.reactions[i].reactants = reac_list[i]
    
    avg_yield_full = avg_yield
    not_removed = set()
    remaining_specs = ordered_species[spec_len-red_mech_size:]
    
    remove_species = []
    for i in ordered_species[:spec_len-red_mech_size]:
        if i not in protected:
            remove_species.append(i)
        else:
            remaining_specs.append(i)

    # the scc's in which at least one species is remaining
    remaining_scc = set()
    for i in remaining_specs:
        if i in scc_set:
            remaining_scc.add(scc_dict_2[i])

    removed_scc = set(range(len(scc))).difference(remaining_scc)
    scc_remove_species = set()

    for i in removed_scc:
        for j in scc[i]:
            scc_remove_species.add(j)

    if settings['Print Progress'] == True:
        print('Stage 4: Combining Grouped Species')

    # in this section, grouped species are combined together with the remaining species in the group with the highest yield/priority
    new_groups = []
    man_group_set = set()
    # adding in any groups supplied by the user and overwriting any groups that contained species now in a manual group
    for i in manual_groups:
        for j in i:
            man_group_set.add(j)
    for i in groups:
        new_group = []
        for j in i:
            if j not in man_group_set:
                new_group.append(j)
        if len(new_group)>1:
            new_groups.append(new_group)
    groups= deepcopy(new_groups)
    new_groups = []
    for i in groups:
        if not all([x in remaining_specs for x in i]):
            new_groups.append(i)
    groups = deepcopy(new_groups)
    

    
    # modifying the groups in regard to scc's
    # if a group is comprised of species from two or more separate scc's:
    # the group will be merged if and only if no species remain in the respective scc's of the merged species
    # this prevents the creation of new scc's or the expansion of existing scc's
    # does not apply to manually added groups
    new_groups = []
    for i in range(len(groups)):
        no_scc_group = []
        scc_groups = {}
        none_left = []
        for j in groups[i]:
            if j not in no_group:
                if j in scc_set:
                    if scc_dict_2[j] in scc_groups:
                        scc_groups[scc_dict_2[j]].append(j)
                    else:
                        scc_groups[scc_dict_2[j]] = [j]
                else:
                    no_scc_group.append(j)
        for g in scc_groups:
            scc_ind = scc_dict_2[scc_groups[g][0]]
 
            scc_specs = []
            for y in scc[scc_ind]:
                if y not in scc_groups[g]:
                    scc_specs.append(y)

            counted = 0

            if not any([x in remaining_specs for x in scc_specs]):

                for h in scc_groups:
                    if g!=h:
                        if any([x in remaining_specs for x in scc_groups[h]]) and counted==0:
                            counted+=1
                            
                            for f in scc_groups[g]:
                                scc_groups[h].append(f)
                                if f in remaining_specs:
                                    
                                    remaining_specs.remove(f)
                                    remove_species.append(f)
                if counted==0:
                    for f in scc_groups[g]:
                        none_left.append(f)
                scc_groups[g] = []
                        
        if len(no_scc_group)>1:
            new_groups.append(no_scc_group)
        for k in scc_groups:
            if len(scc_groups[k])>1:
                new_groups.append(scc_groups[k])
        if len(none_left)>1:
            new_groups.append(none_left)
    groups = deepcopy(new_groups)   
    
    for m in manual_groups:
        if any([x in remaining_specs for x in m]):
            if any([x in remove_species for x in m]):
                groups.append(m)     
    group_set = set()
    grouped = set()
    group_dict = {}
    rem_groups = []
    count = 0

    # selecting only groups in which at least one species is removed and one species remains
    for i in groups:
        if any([j in remaining_specs for j in i]) and any([k in remove_species for k in i]):
            rem_groups.append(i)
            for j in i:
                group_set.add(j)
                group_dict[j] = count
            count+=1
    groups = deepcopy(rem_groups)
    # group_mapping: a map connecting species in the full mechanism to merged species in the reduced mechanism
    group_mapping = []
    
    #dictionary to map grouped species to group
    group_dict_check = {}
    
    # merging grouped species begins here
    for j in range(len(groups)):
        mapping = [0]
        group = groups[j]
        # identifying the species in the group to merge into
        for i in ordered_species[spec_len-red_mech_size:]:
            if i in group:
                group_lead = i
                mapping[0] = group_lead
        out_reactions = {}
        removers = set()

        # removing any species from the group that are not being removed or merged into
        new_group = {group_lead}
        for i in group:
            if i not in remove_species:
                pass

            else:
                new_group.add(i)
        group = deepcopy(new_group) 
        for sp in new_group:
            group_dict_check[sp] = new_group
        total_yield = 0 
        grouped_map = []
        for i in group:
            grouped.add(i)
            grouped_map.append(i)
            total_yield += avg_yield_o2[i]
        # total_yield: the total yield of the group
        mapping.append(grouped_map)
        group_mapping.append(mapping)

        
        
        new_graph_gl = {}
        within_spec_ratio = {}
        
        #mult * = base * spec_rat * rate_type[x]/rate_sum  /rate 
        for i in group:
            for r in rxn_reac[i]:
                if any([x in group for x in reduced_mech.reactions[r].prod_dict]):
                    new_prod_dict = {}
                    
                    for p in reduced_mech.reactions[r].prod_dict:
                        if p in group:
                            if p in reduced_graph[i]:
                                del reduced_graph[i][p]
                        else:
                            new_prod_dict[p] = reduced_mech.reactions[r].prod_dict[p]
                        
                    reduced_mech.reactions[r].prod_dict = deepcopy(new_prod_dict)                      

        # the faction of group yield coming from the group lead
        base_spec = avg_yield_o2[group_lead]/max(1e-20,total_yield)
        base_rate_type = {}
        for g in groups[j]:
            for x in out_graph_type[g]:
                for y in out_graph_type[g][x]:
                    base_rate_type[y] = 0
        

        # base_rate_type: a dictionary containing the relative rate of reaction of each reaction type that the group lead participates in
        base_rate = 0
        for r in rxn_reac[group_lead]:
            if rxn_types[r] in base_rate_type:
                base_rate_type[rxn_types[r]]+= np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier
            else:
                pass
                #print('no match', groups[j], reduced_mech.reactions[r].reactants)
            base_rate+=np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier

        # the multiplying factor for reactions involving the group lead
        for r in rxn_reac[group_lead]:
            
            mult = base_spec

            reduced_mech.reactions[r].multiplier *= mult
        
        for i in group:
            if i != group_lead:
                # i is the current species being merged
                # the fraction of group yield coming from i
                spec_rat = avg_yield_o[i]/max(1e-20,total_yield)
                rate_type = {}
                for x in out_graph_type[i]:
                    for y in out_graph_type[i][x]:
                        rate_type[y] = 0
                rate_sum = 0
                for r in rxn_reac[i]:
                    rate_type[rxn_types[r]]+= np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier
                    rate_sum+=np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier
                # wherever the removed species appears as a reactant, it is replaced with the group lead
                # the reaction rate is adjusted proportionally to its yield relative to the overall group, as defined by mult
                for r in rxn_reac[i]:
                    if rate_sum>0:
                        mult = spec_rat*base_rate/rate_sum
                        
                    else:
                        mult = 0
                    new_reactants = []
                    for x in reduced_mech.reactions[r].reactants:
                        if x in group:
                            new_reactants.append(group_lead)
                        else:
                            new_reactants.append(x)
                    reduced_mech.reactions[r].reactants = new_reactants
                    reduced_mech.reactions[r].multiplier*=mult
                    rxn_reac[group_lead].add(r)

                    for p in reduced_mech.reactions[r].prod_dict:
                        
                        reduced_out[group_lead].add(p)
                        reduced_in[p].add(group_lead)
                # wherever the removed species appears as a product, it is replaced with the group lead and the reduced graph is adjusted as well
                for r in rxn_prod[i]:
                    if i in reduced_mech.reactions[r].prod_dict:
                        if group_lead in reduced_mech.reactions[r].prod_dict:
                            reduced_mech.reactions[r].prod_dict[group_lead] += reduced_mech.reactions[r].prod_dict[i]
                        else:
                            reduced_mech.reactions[r].prod_dict[group_lead] = reduced_mech.reactions[r].prod_dict[i]
                        del reduced_mech.reactions[r].prod_dict[i]
                        rxn_prod[group_lead].add(r)
                        for re in reduced_mech.reactions[r].reactants:
                            if re not in background_spc:
                                if i in reduced_graph[re]:
                                    if group_lead in reduced_graph[re]:
                                        reduced_graph[re][group_lead]+= reduced_graph[re][i]
                                    else:
                                        reduced_graph[re][group_lead] = reduced_graph[re][i]
                for x in reduced_out[i]:
                    if i in reduced_in[x]:
                        reduced_in[x].remove(i)
                for x in reduced_in[i]:
                    if i in reduced_out[x]:
                        reduced_out[x].remove(i)
                    if i in reduced_graph[x]:
                        del reduced_graph[x][i]
        # updating the reduced graph for the group lead edges to reflect changes as a result of the merge1WS
        total_rate = 0
        new_in_gl = set()
        for r in rxn_prod[group_lead]:
            for j in reduced_mech.reactions[r].reactants:
                if j not in background_spc:
                    new_in_gl.add(j)
                    
        for r in rxn_reac[group_lead]:
            total_rate+= np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier
        new_out_gl = set()
        for r in rxn_reac[group_lead]:
            for p in reduced_mech.reactions[r].prod_dict:
                new_out_gl.add(p)
                if p in new_graph_gl:
                    new_graph_gl[p]+= reduced_mech.reactions[r].prod_dict[p]*np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier/max(1e-20,total_rate)
                else:
                    new_graph_gl[p] = reduced_mech.reactions[r].prod_dict[p]*np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier/max(1e-20,total_rate)
        reduced_graph[group_lead] = new_graph_gl
        reduced_out[group_lead] = new_out_gl
        reduced_in[group_lead] = new_in_gl

    if settings['Print Progress'] == True:
        print('Stage 5: Converting Graph to Directed Acyclic Graph')

    # Here species in strongly connected components are removed assuming at least one species in the SCC remains.
    removed_rxns = set()
    dont_remove = set()
    for i in remaining_specs:
        dont_remove.add(i)
    for i in grouped:
        dont_remove.add(i)
    scc_rem_removed = set()
    for i in remaining_scc:
        just_removed = set()
        for j in scc[i]:
            if j not in dont_remove:
                for r in rxn_reac[j]:
                    removed_rxns.add(r)
                #here we modify the graph
                for ins in reduced_in[j]:
                    # if a species is both a parent and a child of the removed species, it will not be rerouted to itself, because there are no self edges
                    # scale factor is multiplier for the remaining products to maintain a molar balance
                    # when rerouting an scc species, the products of the removed species are added to the products of the parent species
                    # in proportion of the yield of the removed species from the parent species
                    # if the parent species is in the scc, then the edges are rerouted according to the standard graph
                    # if the parent species is not in the scc, then we reroute edges using the DAG representation
                    # this is done to avoid adding new species into the scc which would mess with the mechanism dynamics
                    if ins in reduced_graph[j]:
                        scale_factor = abs(1/max(1e-20,1-reduced_graph[j][ins]))
                    else:
                        scale_factor = 1
                    
                    if j in reduced_graph[ins]: 
                        mark = False
                        # here we are checking to see if the parent species we are rerouting to is in the scc, or if it is in a group with a species in the scc
                        if ins in group_dict_check:
                            if any([t in scc[i] for t in group_dict_check[ins]]):
                                mark = True
                        # if yes to the above, we use the non-dag graph to reroute the edge
                        if ins in scc[i] or mark==True:
                            for sp in reduced_graph[j]:
                                if sp!= ins:
                                    if sp in reduced_graph[ins]:
                                        
                                        reduced_graph[ins][sp]+= reduced_graph[j][sp]*reduced_graph[ins][j]*scale_factor
                                    else:
                                        reduced_graph[ins][sp] = reduced_graph[j][sp]*reduced_graph[ins][j]*scale_factor
                                    reduced_out[ins].add(sp)
                                    reduced_in[sp].add(ins)
                        # if no to the above, we use the dag graph to reroute the edge
                        # average_cycle_out_graph is a dag representation of the products from a given scc species
                        else:
                            for sp in avg_cycle_out_graph[j]:
 
                                if sp in reduced_graph[ins]:
                                    reduced_graph[ins][sp]+= avg_cycle_out_graph[j][sp]*reduced_graph[ins][j]
                                else:

                                    reduced_graph[ins][sp] = avg_cycle_out_graph[j][sp]*reduced_graph[ins][j]
                                reduced_out[ins].add(sp)
                                reduced_in[sp].add(ins)
            
                            
                                
                    reduced_out[ins].remove(j)
                    del reduced_graph[ins][j]
                # here we modify the reactions
                for r in rxn_prod[j]:
                    if j in reduced_mech.reactions[r].prod_dict:
                        
                        mark = False
                        for spe in reduced_mech.reactions[r].reactants:
                            if spe in group_dict_check:
                                if any([t in scc[i] for t in group_dict_check[spe]]):
                                    mark = True
                        mark = True
                        if any([f in scc[i] for f in reduced_mech.reactions[r].reactants]) or mark==True:
                            scale_factor = 1
                            for rec in reduced_mech.reactions[r].reactants:
                                if rec in scc[i]:
                                    if rec in reduced_graph[j]:
                                        scale_factor = abs(1/max(1e-20,1-reduced_graph[j][rec]))
                            for pro in reduced_graph[j]:
                                if pro not in reduced_mech.reactions[r].reactants:
                                    if pro in reduced_mech.reactions[r].prod_dict:
                                        reduced_mech.reactions[r].prod_dict[pro] += reduced_graph[j][pro]*reduced_mech.reactions[r].prod_dict[j]*scale_factor
                                    else:
                                        reduced_mech.reactions[r].prod_dict[pro] = reduced_graph[j][pro]*reduced_mech.reactions[r].prod_dict[j]*scale_factor
                                    
                                    rxn_prod[pro].add(r)
                            
                        else:
                            for pro in avg_cycle_out_graph[j]:
                                if pro not in reduced_mech.reactions[r].reactants:
                                    if pro in reduced_mech.reactions[r].prod_dict:
                                        reduced_mech.reactions[r].prod_dict[pro] += avg_cycle_out_graph[j][pro]*reduced_mech.reactions[r].prod_dict[j]
                                    else:
                                        reduced_mech.reactions[r].prod_dict[pro] = avg_cycle_out_graph[j][pro]*reduced_mech.reactions[r].prod_dict[j]
                                    rxn_prod[pro].add(r)
                        del reduced_mech.reactions[r].prod_dict[j]
                scc_rem_removed.add(j)  

    
    # here we are creating the full dag
    # dag_graph, dag_in, and dag_out have the same format as reduced_graph, reduced_in, and reduced_out, but are directed acyclic graphs
    marked = set(background_spc).union(remaining_specs)
    dag_graph = [{} for i in range(spec_len)]
    dag_out = [set() for i in range(spec_len)]
    dag_in = [set() for i in range(spec_len)]
    count = 0
    
    for i in in_cycle_specs:
        for j in i:
            dicy = {}
            for k in avg_cycle_out_graph[j]:
                if avg_cycle_out_graph[j][k]!=0:
                    dicy[k]=avg_cycle_out_graph[j][k]
                    dag_out[j].add(k)
                    dag_in[k].add(j)
            dag_graph[j] = dicy
    for i in range(spec_len):
        count+=1
        if i not in scc_set:
            dicy = {}
            for j in reduced_graph[i]:
                if reduced_graph[i][j]!=0:
                    dicy[j]=reduced_graph[i][j]
                    dag_out[i].add(j)
                    dag_in[j].add(i)
            dag_graph[i] = dicy

    remove_these_rxns = set()
    avg_cycle_out_graph_2 = {}
    for i in avg_cycle_out_graph:
        dicy = {}
        for j in avg_cycle_out_graph[i]:
            if avg_cycle_out_graph[i][j]!=0:
                dicy[j]= avg_cycle_out_graph[i][j]
        avg_cycle_out_graph_2[i] = dicy

    for i in range(len(reduced_graph)):
        dicy = {}
        for j in reduced_graph[i]:
            if reduced_graph[i][j]!=0:
                dicy[j]= reduced_graph[i][j]
        reduced_graph[i]=dicy

    
    if settings['Print Progress'] == True:
        print('Stage 6: Creating Category Species')
    
    # here we are using the categorization method for species removal
    # every species placed into the set of user-specified categories will be removed (if it not in the final remaining species)
    # its reactions will be added to a larger set of reactions representing the categorical species
    # each category will have one new species to represent it. 
    new_dict = deepcopy(dic)
    
    categories = []
    cat_names = []
    new_species = deepcopy(mechanism.species)
    cat_species_indices = []
    for i in settings['Categories']:
        caty = []
        for j in i[1]:
            caty.append(dic[j])
        categories.append(caty)
        cat_names.append(i[0])
        new_species.append(i[0])
        new_dict[i[0]] = len(new_species)-1
        cat_species_indices.append(len(new_species)-1)
    cat_dict = {}
    cat_set = set()
    count = 0
    for i in categories:
        for j in i:
            cat_dict[j] = count
            cat_set.add(j)
        count+=1
    cat_specs = []
    # this is the mapping of removed species in the full mechanism to their respective category species in the reduced mechanism
    cat_mapping = []
    
    #category rate multiplier: do the exact same method as the groups 
    #Basically, take the rate of reaction, multiply it by the % contribution of the species
    # and the % contribution of the rxn type compared to a base rate which is selected as the highest
    # contributor to the category
    # if a product in a reaction is in the category, what ever the moles to in-cat is the fraction
    # within cat, and the rate should be multiplied by 1 - frac, and all products should be multiplied 
    # by 1/(1-frac) except for the within cat which is removed
    all_cat_set = set()
    for i in categories:
        for j in i:
            all_cat_set.add(j)
    multses = []
    mult_list_list = []
    cat_data = []
    multy = []
    
    for j in range(len(categories)):
        mapping = []
        # creating the new category species
        spec_len = spec_len+1
        dag_out.append(set())
        dag_in.append(set())
        dag_graph.append({})
        new_spec = cat_species_indices[j]
        mapping.append(new_spec)
        cat_specs.append(new_spec)
        all_cat_set.add(new_spec)
        caty = categories[j]
        reduced_mech.species.append(cat_names[j])
        removers = set()
        rxy_prod = set()
        rxy_reac = set()
        net_yield = 0
        net_out_yield = 0
        out_reactions = {}
        multies = []
        data_c = []
        # removing any species from the category that were already removed via grouping or that will remain in the mechanism
        for i in caty:
            
            if i in remaining_specs:
                removers.add(i)
                
            if i in grouped:
                removers.add(i)

        for i in removers:
            caty.remove(i)
        cat_map = []
        caty_yields = {}
        caty_yields_all = {}
        caty_yields_cyc = {}
        caty_yields_cyc_all = {}
        max_yi = 0
        if len(caty)==0:
            break
        # calculating yield of the category, including only mass coming in from outside of the category 
        # mass transfer between two species within the category is not counted
        for i in caty:
            spec_yi = 0
            spec_yi_c = 0
            for x in dag_in[i]:
                if x not in caty:
                    if i in dag_graph[x]:
                        
                        spec_yi += dag_graph[x][i]*avg_yield_o[x]
                        if x in avg_yield_o:
                            spec_yi_c += dag_graph[x][i]*avg_yield_o[x]
                        
            caty_yields[i] = spec_yi
            caty_yields_cyc[i] = spec_yi_c
            caty_yields_all[i] = avg_yield_o[i]
            caty_yields_cyc_all[i] = avg_yield_o[i]
            
            if spec_yi>max_yi:
                max_yi = deepcopy(spec_yi)
                cat_lead = i
        cat_full_yield = sum(caty_yields.values())
        if settings['Print Progress'] == True:
            print('Category Yield ' + str(j) + ': ', cat_full_yield)#, cat_full_yield_all, cat_full_yield_cyc, cat_full_yield_cyc_all)
        cat_rel_rate = 0
        avg_yield_full[new_spec] = cat_full_yield
        avg_yield_o[new_spec] = cat_full_yield
        cat_rel_out = 0
        cat_weight_list = []
        cat_num_list = []
        # calculating the rate of reaction of species in the category. 
        # If a product of the category species is in the category, this species production is discounted from the rate via the variable delta
        for i in caty:
            rates_c = []
            for r in rxn_reac[i]:
                delta = 1
                for x in reduced_mech.reactions[r].prod_dict:
                    if x in caty:
                        delta = delta - reduced_mech.reactions[r].prod_dict[x]
                cat_rel_rate+=np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier*caty_yields[i]/cat_full_yield
                cat_rel_out+= np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier*max(0,delta)*caty_yields[i]/cat_full_yield
                cat_weight_list.append(caty_yields[i]/max(1e-20,cat_full_yield))
                cat_num_list.append(np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier*max(0,delta))
                
                rates_c.append([i,r, reduced_mech.reactions[r].reactants,np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier])
            data_c.append(rates_c)
        #cat_rel_rate = cat_rel_rate/len(caty)
        #cat_rel_out = cat_rel_out/len(caty)
        cdata = pd.DataFrame({'num':cat_num_list,'weight':cat_weight_list})
        # the total relative rate of reaction for the new category species will be equal to the yield weighted median of the relative rates calculated above
        # the weighted median was the optimal selection based on our testing because it covers against outlier values, where the mean does not
        if len(cdata)>0:
            cat_median_rate = weighted_median(cdata,'num', 'weight')
        else:
            cat_median_rate = 1e-5
        data_c.append(cat_rel_rate)
        mult_list = []
        for i in caty:
            # the ratio of the yield of the removed species to the overall category yield
            spec_yield_frac = caty_yields[i]/max(1e-20,cat_full_yield)
            species_rate = 0
            for r in rxn_reac[i]:
                species_rate+= np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier
            if species_rate>0:  
                multies.append([i, spec_yield_frac, cat_rel_rate, species_rate])
                # multiplier for the reactions involving the removed species to be incorporated into the new category
                # the multiplier is in proportion to the fractional yield of the species relative to the category
                # with a scale factor adjusting for the relative rate of reaction of the species versus the relative rate of the category
                mult = spec_yield_frac*(cat_median_rate)/species_rate
                mult_list.append([i,mult, cat_rel_rate,species_rate, spec_yield_frac])
                
            else:
                mult = 0            
            cat_map.append(i)
            # adjusting the graph to reflect the change
            for k in dag_in[i]:
                dag_out[k].add(new_spec)
                dag_in[new_spec].add(k)
                if i in dag_graph[k]:
                    if new_spec not in dag_graph[k]:
                        dag_graph[k][new_spec] = dag_graph[k][i]
                    else:
                        dag_graph[k][new_spec] += dag_graph[k][i]
                else:
                    pass
                    #print('matchning issue', k, i)
                if i in dag_out[k]:
                    dag_out[k].remove(i)
                    if i in dag_graph[k]:    
                        del dag_graph[k][i]
                    else:
                        pass
            skip_r = set()
            # adjusting the reactions to reflect the change
            # where the removed species is a product, it is replaced with the new category species, assuming the reactants are not also in the category
            for r in rxn_prod[i]:
                
                
                if not any([x in caty for x in reduced_mech.reactions[r].reactants]) and r not in skip_r:
                    

                    rxy_prod.add(r)
                    
                    if i in reduced_mech.reactions[r].prod_dict:
                        if new_spec in reduced_mech.reactions[r].prod_dict:
                            reduced_mech.reactions[r].prod_dict[new_spec] += reduced_mech.reactions[r].prod_dict[i]
                        else:
                            reduced_mech.reactions[r].prod_dict[new_spec] = reduced_mech.reactions[r].prod_dict[i]
                else:
                    remove_these_rxns.add(r)
            # where the removed species is a reactant, the new category species is the new reactant
            # any species within the category are removed from the products
            # the rate constant is adjusted based on the multiplier calculated above
            # delta is 1 minus the sum of stoichiometric coefficients of products within the category
            # these products are removed from the reaction and the stoichiometric coefficients of other products are adjusted upward 
            # (by no more than a factor of 3)
            for r in rxn_reac[i]:
                if any([x not in caty for x in reduced_mech.reactions[r].prod_dict]) and r not in skip_r:
                    delta = 1
                     
                    for x in reduced_mech.reactions[r].prod_dict:
                        if x in caty:
                            delta = delta - reduced_mech.reactions[r].prod_dict[x]
                    if delta>0:
                        out_reactions[r] = delta*avg_yield_full[i]
                        net_out_yield = delta*avg_yield_full[i]
                        rxy_reac.add(r)
                        reduced_mech.reactions[r].multiplier *= mult*delta
                        new_prods = {}
                        p_count = 0
                        for p in reduced_mech.reactions[r].prod_dict:
                            if p not in caty:
                                p_count+=reduced_mech.reactions[r].prod_dict[p]

                        for p in reduced_mech.reactions[r].prod_dict:
                            if p not in caty:
                                new_prods[p] = reduced_mech.reactions[r].prod_dict[p] + (1 - delta)*(reduced_mech.reactions[r].prod_dict[p]/max(p_count,1e-20))
                            if (1/delta)*reduced_mech.reactions[r].prod_dict[p]>3:
                                #print('maxed it')
                                pass
                        # special addition for lost carbon species in gecko mechanisms only
                        if p_count == 0 and 'XCLOST' in species_list_names:
                                
                            if dic['XCLOST'] in new_prods:
                                new_prods[dic['XCLOST']] += 1-delta
                            else:
                                new_prods[dic['XCLOST']] = 1-delta
                        reduced_mech.reactions[r].prod_dict = new_prods
                    else:
                        remove_these_rxns.add(r)
                    reduced_mech.reactions[r].reactants.append(new_spec)
                    if i in reduced_mech.reactions[r].reactants:
                        reduced_mech.reactions[r].reactants.remove(i)
                # if none of the products are in the category, the new reaction is easier to create
                elif all([x not in caty for x in reduced_mech.reactions[r].prod_dict]) and r not in skip_r:
                    out_reactions[r] = avg_yield_full[i]
                    net_out_yield = avg_yield_full[i]
                    rxy_reac.add(r)
                    reduced_mech.reactions[r].multiplier *= mult
                    multy.append([r, reduced_mech.reactions[r].multiplier, mult])
                    new_prods = {}
                    if i in scc_set:
                        new_prods = dag_graph[i]
                        reduced_mech.reactions[r].prod_dict = new_prods
                    reduced_mech.reactions[r].reactants.append(new_spec)
                    if i in reduced_mech.reactions[r].reactants:
                        reduced_mech.reactions[r].reactants.remove(i)
                else:
                    
                    for p in reduced_mech.reactions[r].prod_dict:
                        if r in rxn_prod[p]:
                            rxn_prod[p].remove(r)
                    remove_these_rxns.add(r)
        mult_list_list.append(mult_list)
        mapping.append(cat_map)
        net_out_graph = {}
        multses.append(multies)
        # out reactions + graph outs
        tot_cat_rate = 0
        for r in out_reactions:
            tot_cat_rate += np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier
        for r in out_reactions:
            r_eff = np.mean(reduced_mech.reactions[r].rate)*reduced_mech.reactions[r].multiplier/max(1e-20,tot_cat_rate)
            for p in reduced_mech.reactions[r].prod_dict:
                if p in net_out_graph:
                    net_out_graph[p]+= r_eff*reduced_mech.reactions[r].prod_dict[p]
                else:
                    net_out_graph[p] = r_eff*reduced_mech.reactions[r].prod_dict[p]
        # removing species from the graph
        for i in caty:
            for k in dag_out[i]:
                if k not in caty:
                    dag_in[k].add(new_spec)
                    dag_out[new_spec].add(k)
                    if i in dag_in[k]:
                        dag_in[k].remove(i)
            for k in dag_in[i]:
                if i in dag_graph[k]:
                    del dag_graph[k][i]
                if i in dag_out[k]:
                    dag_out[k].remove(i)
            dag_in[i] = set()
        dag_graph[new_spec] = net_out_graph
        rxn_reac.append(rxy_reac)
        rxn_prod.append(rxy_prod)
        cat_mapping.append(mapping)
        cat_data.append(data_c)
  
    if settings['Print Progress'] == True:
        print('Stage 7: Creating Species Removal Tiers')

    # This is the start of the standard species removal method, wherein a non-grouped, non-category species is removed  
    # scc species in which the full scc is removed are included
    # its edges are rerouted so that the parent species connect to the child species
    # For every rerouting, the number of edges in the mechanism changes by the following amount: p*c - p - c
    # where p is the number of edges incoming to the removed species, ie parent species
    # for large mechanisms, removing species in an arbitrary order can lead to an explosion in the total number of edges
    # to mitigate this, we sort the species into tiers such that p + c >= p*c for all species at the point of their removal
    # this is accomplished by removing species with zero or one outgoing edges first. After those species have been removed
    # new species will have one or zero outgoing edges, which are removed next
    # the following code creates these tiers

    group_set = set()
    group_dict = {}
    for i in range(len(groups)):
        for j in groups[i]:
            group_dict[j] = i
            group_set.add(j)
    remaining_scc = set()
    removed_scc = set()
    remaining_groups = set()
    removed_groups = set()
    group_lead = {}
    reduced_mech_1 = deepcopy(reduced_mech)
    
    for i in cat_specs:
        remaining_specs.append(i)
    
    for i in remaining_specs:
        if i in scc_set:
            remaining_scc.add(scc_dict_2[i])
        if i in group_set:
            remaining_groups.add(group_dict[i])
            group_lead[group_dict[i]] = i
    for i in grouped:
        if i in scc_set:
            remaining_scc.add(scc_dict_2[i])
    removed_scc = set(range(len(scc))).difference(remaining_scc)
    removed_groups = set(range(len(groups))).difference(remaining_groups)

    # this is the variable where the species tiers are stored. it is also given as an output from the algorithm
    tiers = [[]]
    cat_set = set()
    group_set = set()
    tier_set = set(background_spc).union(remaining_specs)
    for i in grouped:
        tier_set.add(i)

    delta = 1
 
    in_cycle_spec_set = set()
    for i in in_cycle_specs:
        for j in i:
            in_cycle_spec_set.add(j)
    remaining_search = {i for i in range(spec_len) if i not in tier_set}
    for i in scc_remove_species:
        if i not in in_cycle_spec_set:
            if i in remaining_search:
                remaining_search.remove(i)
    for i in cat_set:
        if i not in not_removed and i in remaining_search:
            remaining_search.remove(i)
    for i in group_set:
        if i not in not_removed and i in remaining_search:
            remaining_search.remove(i)
    scc_remove_no_in = set()
    for i in scc_remove_species:
        if i not in in_cycle_spec_set:
            scc_remove_no_in.add(i)
    

    remove_1 = deepcopy(remove_these_rxns)
    
    while delta>0:
        new_to_add = set()
        tier = set()
        to_remove = set()
        for j in remaining_search:
            if all([x in tier_set for x in dag_out[j]]):
                new_to_add.add(j)
                to_remove.add(j)
                if j not in cat_set and j not in group_set and j not in scc_remove_no_in:
                    tier.add(j)
        for p in to_remove:
            remaining_search.remove(p)
        delta = len(tier)
        tiers.append(tier)
        tier_set = tier_set.union(new_to_add)
    if len(remaining_search)>0:
        tiers.append(remaining_search)
    
    dag_graph_save = deepcopy(dag_graph)
    
   
    count = 0
    dag_graph_save = deepcopy(dag_graph)

    dag_graph_save_2 = deepcopy(dag_graph)

    remove_2 = deepcopy(remove_these_rxns)
    normal_remove = deepcopy(remove_species)
    normal_remove = set(range(spec_len))
    removed_species = set()
    true_count = 0
    false_count =0
    remaining_specs = set(remaining_specs)


    remove_3 = deepcopy(remove_these_rxns)
    
    if settings['Print Progress'] == True:
        print('Stage 8: Removing Remaining Species')

    reduced_mech_1 = deepcopy(reduced_mech)
    # here all species are removed in order of the tiers
    for t in tiers:
        for i in t:
            # here we have scc species in which the entire SCC is going to be removed
            # because we are looking at the DAG, only SCC species that originally had parent species outside of the DAG will remain in the DAG
            if i in scc_set:
                scc_indy = scc_dict_2[i]
                if i not in grouped:
                    count+=1
                    if count%1000==0:
                        pass
                        #print(count)
                    # modifying graph
                    for k in dag_in[i]:
                        if k not in scc[scc_indy]:
                            if i in dag_graph[k]:
                                mult = dag_graph[k][i]
                            else:
                                mult = 0
                            
                            for p in dag_graph[i]:
                                dag_in[p].add(k)
                                dag_out[k].add(p)
                                if p not in dag_graph[k]:
                                    dag_graph[k][p] = mult*dag_graph[i][p]
                                else:
                                    dag_graph[k][p] += mult*dag_graph[i][p]
    
                            if i in dag_out[k]:
                                dag_out[k].remove(i)
                            if i in dag_graph[k]:
                                del dag_graph[k][i]    

                    # modifying reactions
                    for r in rxn_prod[i]:
                        if r not in removed_rxns:
                            if not any([x in scc[scc_indy] for x in reduced_mech.reactions[r].reactants]):
                                if i in reduced_mech.reactions[r].prod_dict:
                                    mult = reduced_mech.reactions[r].prod_dict[i]
                                else:
                                    mult = 0
                                for p in dag_graph[i]:
                                    if p not in reduced_mech.reactions[r].reactants and p not in scc[scc_indy]:
                                        if p in reduced_mech.reactions[r].prod_dict:
                                            reduced_mech.reactions[r].prod_dict[p]+= mult*dag_graph[i][p]
                                        else:
                                            reduced_mech.reactions[r].prod_dict[p] = mult*dag_graph[i][p]
                                        rxn_prod[p].add(r)
                    for r in rxn_reac[i]:
                        removed_rxns.add(r)
            # species not in an scc
            else:

                removed_species.add(i)
                count+=1
                # modifying graph
                removers = set()
                for j in dag_in[i]:
                    if i in dag_out[j]:
                        dag_out[j].remove(i)
                    if i in dag_graph[j]:
                        mult = dag_graph[j][i]
                        del dag_graph[j][i]
                    else:
                        mult = 0
    
                    for k in dag_out[i]:
                        
                        if k!=j and dag_graph[i][k]*mult!=0:
                            dag_out[j].add(k)
                            dag_in[k].add(j)
                            if k in dag_graph[j]:
                                dag_graph[j][k] += dag_graph[i][k]*mult
                            else:
                                dag_graph[j][k] = dag_graph[i][k]*mult

     
                    removers.add(j)
                    
                    
                # modifying mechanism
                if any([x in remaining_specs for x in dag_in[i]]):
                    true_count+=1
                    
                    for r in rxn_prod[i]:
    
                        if r not in removed_rxns:
    
                            if i in reduced_mech.reactions[r].prod_dict:
                                mult = reduced_mech.reactions[r].prod_dict[i]
                            else:
                                mult = 0
                            for j in dag_graph[i]:
    
                                if j not in reduced_mech.reactions[r].reactants and mult*dag_graph[i][j]!=0:
    
                                    rxn_prod[j].add(r)
                                    if j in reduced_mech.reactions[r].prod_dict:
                                        reduced_mech.reactions[r].prod_dict[j] += mult*dag_graph[i][j]
                                    else:
                                        reduced_mech.reactions[r].prod_dict[j] = mult*dag_graph[i][j]
                    for r in rxn_reac[i]:
    
                        removed_rxns.add(r)
                for p in removers:
                    dag_in[i].remove(p)



    reduced_mech_3 = deepcopy(reduced_mech)
    
    remove_4 = deepcopy(remove_these_rxns)


    if settings['Print Progress'] == True:
        print('Stage 9: Mechanism Clean Up')
    # here we remove unneeded reactions
    remove_set = set(remove_species)
    back_set_set = set(back_set)
    full_set = remove_set.union(back_set_set)
    
    for i in range(len(reduced_mech.reactions)):
        
        for j in reduced_mech.reactions[i].reactants:
            if j in remove_set:
                remove_these_rxns.add(i)

    new_reactions = []
    count = 0
    for i in range(len(reduced_mech.reactions)):
        if i not in remove_these_rxns:
            
            new_reactions.append(deepcopy(reduced_mech.reactions[i]))
            count+=1


    remove_species = set(remove_species)

    for i in range(len(new_reactions)):

        new_reactants = deepcopy(new_reactions[i].reactants)
        new_prod_dict = {}
        for j in new_reactions[i].prod_dict:
            if new_reactions[i].prod_dict[j]>0.001 and j not in remove_species:
                # any product stoichiometric coefficient less than 0.001 is removed
                # FLAG THIS 0.001 value
                new_prod_dict[j] = new_reactions[i].prod_dict[j]
        new_reactions[i].prod_dict = new_prod_dict



    new_new_reactions = []
    reduced_mech_1 = deepcopy(reduced_mech)
    for i in new_reactions:
        if i.multiplier>0:
            new_new_reactions.append(i)

    new_reactions = deepcopy(new_new_reactions)
    
    remove_5 = deepcopy(remove_these_rxns)

    reduced_mech.reactions = deepcopy(new_reactions)
   

    # consolidating full to reduced mechanism species mapping
    accounted_specs = set()
    new_cat_mapping = []
    map_dict = {}
    for i in background_spc:
        map_dict[i] = i
    for i in cat_mapping:
        if len(i[1])>0:
            accounted_specs.add(i[0])
            new_cat_mapping.append(i)
            for k in i[1]:
                map_dict[k] = i[0]
    new_group_mapping = []
    group_map_dict = {}
    for i in group_mapping:
        if len(i[1])>0:
            accounted_specs.add(i[0])
            new_group_mapping.append(i)
            group_map_dict[i[0]] = i[1]
            for k in i[1]:
                map_dict[k] = i[0]
        map_dict[i[0]] = i[0]
            
    normal_specs = []
    for i in remaining_specs:
        if i not in accounted_specs:
            normal_specs.append(i)
            map_dict[i] = i
    represented_specs = []
    for i in new_group_mapping:
        represented_specs.append(i[0])
        for j in i[1]:
            represented_specs.append(j)
    for i in new_cat_mapping:
        for j in i[1]:
            represented_specs.append(j)
    for i in normal_specs:
        represented_specs.append(i)
    full_map = [new_group_mapping, new_cat_mapping, normal_specs]
    map_yields = [[] for i in range(len(represented_specs))]
    inv_map_dict = {}
    for i in map_dict:
        if map_dict[i] in inv_map_dict:
            inv_map_dict[map_dict[i]].add(i)
        else:
            inv_map_dict[map_dict[i]] = {i}
    represented_set = set(represented_specs)
    
    if settings['Print Progress'] == True:
        print('Stage 10: Analyzing Reduced Mechanism')

    # recreating graph, identifying scc's and measuring yields for the reduced mechanism  
    red_graphs = [[{} for i in range(spec_len)] for c in range(l_c)]
    red_in_graph = [set() for i in range(spec_len)]
    red_out_graph = [set() for i in range(spec_len)]

    
    red_dag_out = []
    new_rxns = []
    for i in reduced_mech.reactions:
        if i.prod_dict!={}:
            new_rxns.append(i)

    reduced_mech.reactions = deepcopy(new_rxns)
    red_specs = []
    for i in new_dict:
        if new_dict[i] in remaining_specs:
            red_specs.append(new_dict[i])

    red_rxn_reac = {}
    red_rxn_prod = {}
    red_spec_dict = {}
    for i in range(len(red_specs)):
        red_spec_dict[red_specs[i]] = i
    for i in range(len(reduced_mech.reactions)):
        for j in reduced_mech.reactions[i].reactants:
            if j not in back_set:
                if j in red_rxn_reac:  
                    red_rxn_reac[j].add(i)
                else:
                    red_rxn_reac[j] = {i}
        for p in reduced_mech.reactions[i].prod_dict:
            if p not in back_set:
                if p in red_rxn_prod:
                    red_rxn_prod[p].add(i)
                else:
                    red_rxn_prod[p] = {i}
    for i in red_rxn_reac:
        if i not in back_set:
            edges = [{} for c in range(l_c)]
            rate_sums = [0 for c in range(l_c)]
            
            for j in red_rxn_reac[i]:

                for c in range(l_c):
                    rate_sums[c] = rate_sums[c] + reduced_mech.reactions[j].rate[c]
            for j in red_rxn_reac[i]:
                mults = [0 for c in range(l_c)]
                for c in range(l_c):
                    mults[c] = reduced_mech.reactions[j].rate[c]/max(1e-20,rate_sums[c])
                if len(reduced_mech.reactions[j].reactants) == 1:
                    type_r = 'solo'
                else:
                    mark = True
                    for p in reduced_mech.reactions[j].reactants:
                        if p in background_spc:
                            type_r = p
                            mark = False
                    if mark:
                        type_r = 'double'
                for k in reduced_mech.reactions[j].prod_dict:
                    
                    for c in range(l_c):
                        if k in edges[c]:
                            if c==0:
                                edges[c][k] = edges[c][k] + reduced_mech.reactions[j].prod_dict[k]*mults[c]
                                red_out_graph[i].add(k)
                                red_in_graph[k].add(i)
                            else:
                                edges[c][k] = edges[c][k] + reduced_mech.reactions[j].prod_dict[k]*mults[c]

                        
                        else:
                            if c==0:
                                edges[c][k] = reduced_mech.reactions[j].prod_dict[k]*mults[c]
                                red_out_graph[i].add(k)
                                red_in_graph[k].add(i)
                            else:
                                edges[c][k] = reduced_mech.reactions[j].prod_dict[k]*mults[c]
            for c in range(l_c):
                red_graphs[c][i] = edges[c]
    
    
    

    for i in red_rxn_reac:
        if i not in back_set:
            edges = [{} for c in range(l_c)]
            rate_sums = [0 for c in range(l_c)]
            
            for j in red_rxn_reac[i]:

                for c in range(l_c):
                    rate_sums[c] = rate_sums[c] + reduced_mech.reactions[j].rate[c]*reduced_mech.reactions[j].multiplier
            for j in red_rxn_reac[i]:
                mults = [0 for c in range(l_c)]
                for c in range(l_c):
                    mults[c] = reduced_mech.reactions[j].multiplier*reduced_mech.reactions[j].rate[c]/max(1e-20,rate_sums[c])
                if len(reduced_mech.reactions[j].reactants) == 1:
                    type_r = 'solo'
                else:
                    mark = True
                    for p in reduced_mech.reactions[j].reactants:
                        if p in background_spc:
                            type_r = p
                            mark = False
                    if mark:
                        type_r = 'double'
                for k in reduced_mech.reactions[j].prod_dict:
                    
                    for c in range(l_c):
                        if k in edges[c]:
                            if c==0:
                                edges[c][k] = edges[c][k] + reduced_mech.reactions[j].prod_dict[k]*mults[c]
                                red_out_graph[i].add(k)
                                red_in_graph[k].add(i)
                            else:
                                edges[c][k] = edges[c][k] + reduced_mech.reactions[j].prod_dict[k]*mults[c]

                        
                        else:
                            if c==0:
                                edges[c][k] = reduced_mech.reactions[j].prod_dict[k]*mults[c]
                                red_out_graph[i].add(k)
                                red_in_graph[k].add(i)
                            else:
                                edges[c][k] = reduced_mech.reactions[j].prod_dict[k]*mults[c]
            for c in range(l_c):
                red_graphs[c][i] = edges[c]
    
    red_graph_mat_lil = lil_matrix((spec_len,spec_len), dtype=None)
    

        
    red_edges = []
    for i in range(len(red_out_graph)):
        for j in red_out_graph[i]:
            red_edges.append((i,j))
    count = 0
    for i,j in red_edges:
        red_graph_mat_lil[i,j] = 1
    reduced_mech_4 = deepcopy(reduced_mech)
    red_graph_mat = red_graph_mat_lil.tocsr(copy=False)

    red_scc_result = connected_components(red_graph_mat, directed=True, connection='strong', return_labels=True)

    red_scc_dic = {}
    for i in range(len(red_scc_result[1])):
        if red_scc_result[1][i] in red_scc_dic:
            red_scc_dic[red_scc_result[1][i]].append(i)
        else:
            red_scc_dic[red_scc_result[1][i]] = [i]
    
    red_scc = []
    
    for i in red_scc_dic:
        if len(red_scc_dic[i])>1:
            if not any([x in cat_specs for x in red_scc_dic[i]]):
                pass
            else:
                c_specs = []
                for x in red_scc_dic[i]:
                    if x in cat_specs:
                        c_specs.append(x)
                not_allowed = set()
                for x in red_scc_dic[i]:
                    if x not in c_specs:
                        not_allowed.add(x)
                for c in c_specs:
                    new_out_g = set()
                    for j in red_out_graph[c]:
                        if j not in not_allowed:
                            new_out_g.add(j)
                        else:
                            if c in red_in_graph[j]:
                                red_in_graph[j].remove(c)
                    red_out_graph[c] = new_out_g

                    for cond in range(l_c):
                        new_graph_c = {}
                        for j in red_graphs[cond][c]:
                            if j not in not_allowed:
                                new_graph_c[j] = red_graphs[cond][c][j]
                        red_graphs[cond][c] = deepcopy(new_graph_c)
                    
                    for r in red_rxn_reac[c]:
                        new_p = {}
                        for p in reduced_mech.reactions[r].prod_dict:
                            if p not in not_allowed:
                                new_p[p] = reduced_mech.reactions[r].prod_dict[p]
                            else:
                                red_rxn_prod[p].remove(r)
                        reduced_mech.reactions[r].prod_dict = new_p

    red_graph_mat_lil = lil_matrix((spec_len,spec_len), dtype=None)
    red_edges = []
    for i in range(len(red_out_graph)):
        for j in red_out_graph[i]:
            red_edges.append((i,j))
    count = 0
    for i,j in red_edges:
        red_graph_mat_lil[i,j] = 1
    reduced_mech_4 = deepcopy(reduced_mech)
    red_graph_mat = red_graph_mat_lil.tocsr(copy=False)

    red_scc_result = connected_components(red_graph_mat, directed=True, connection='strong', return_labels=True)

    red_scc_dic = {}
    for i in range(len(red_scc_result[1])):
        if red_scc_result[1][i] in red_scc_dic:
            red_scc_dic[red_scc_result[1][i]].append(i)
        else:
            red_scc_dic[red_scc_result[1][i]] = [i]
    for i in red_scc_dic:
        if len(red_scc_dic[i])>1:
            red_scc.append(red_scc_dic[i])
            
 

    red_scc_dict_2 = {}
    red_scc_set = set()
    for i in range(len(red_scc)):
        for j in red_scc[i]:
            red_scc_dict_2[j] = i
            red_scc_set.add(j)
    red_scc_map = {}
    for i in red_scc_set:
        red_scc_map[i] = inv_map_dict[i]
    

    if settings['Print Progress'] == True:
        print('Stage 11: Optimizing Cyclical Rate Constants')

    scc_ref_data = {}
    represented_scc = set()
    for i in red_scc_map:
        for j in red_scc_map[i]:
            if j in scc_dict_2:
                represented_scc.add(scc_dict_2[j])
    # this section is used to optimize the rate constants of any reactions involved in scc's in the reduced mechanism
    # does not include scc's involving category species
    
# the yields of the a subset of the full mechanism are calculated to obtain reference values for the optimization
    if settings['Iterations']>0:
        for i in represented_scc:
            ref_data = []
            for c in range(l_c):
                in_specs = []
                for k in in_cycle_specs[i]:
                    k_start = 0
                    for s in in_graph[k]:
                       
                        if s not in scc[i]:
                            if k in graphs[c][s]:
                                k_start+=graphs[c][s][k]*yields[c][s]
                    in_specs.append([k,k_start])
                    
                data, within_data, out_data = cycle_simulator_3_modified(scc[i], in_specs, graphs[c],out_graph,in_graph, [leny+40,2*(leny+40)],1e-6,set(scc[i]).union(scc_out_specs[i]),scc_out_specs[i])
    
                
                new_data = {}
                for d in data:
                    if d not in represented_set:
                        
                        yi = get_yields_modified([d], represented_set, 1e-10,new_graphs[c], new_ins[c], new_outs[c], back_set,scc, scc_dict_2, scc_set)
                        for y in yi:
                            if yi[y]!=0:
                                if y in new_data:
                                    new_data[y]+=data[d]*yi[y]
                                else:
                                    new_data[y]=data[d]*yi[y]
                    else:
                        if d in new_data:
                            new_data[d]+= data[d]
                        else:
                            new_data[d]= data[d]
    
                ref_data.append([new_data, within_data, out_data])
            scc_ref_data[i] = ref_data

    red_dag_out = deepcopy(red_out_graph)
    red_dag_in = deepcopy(red_in_graph)
    xspecs = []
    for i in red_scc:
        xspec = i[0]
        xspecs.append(i[0])
        for j in i:
            if j != xspec:
                for x in red_dag_out[j]:
                    if x not in i:
                        red_dag_out[xspec].add(x)
                        red_dag_in[x].add(xspec)
                    if j in red_dag_in[x]:
                        red_dag_in[x].remove(j)
                for x in red_dag_in[j]:
                    red_dag_out[x].add(xspec)
                    red_dag_in[xspec].add(x)
                    red_dag_out[x].remove(j)
                red_dag_out[j] = set()
                red_dag_in[j] = set()

    for i in range(len(red_dag_in)):
        if i in red_dag_in[i]:
            red_dag_in[i].remove(i)
        if i in red_dag_out[i]:
            red_dag_out[i].remove(i)

    searcher = set(deepcopy(roots))
    dont_visit = set()
    for i in range(len(red_dag_in)):
        if len(red_dag_in[i])==0:
            searcher.add(i)
            dont_visit.add(i)
    # determining which scc to optimize based on proximity to root species
    visited = []
    visit_set = set()
    while len(searcher)>0:
        
        new_searcher = deepcopy(searcher)
        for i in searcher:
            if i not in visit_set:
                if i not in dont_visit:
                    visited.append(i)
                visit_set.add(i)
            new_searcher.remove(i)
            for j in red_dag_out[i]:
                if all([x in visit_set for x in red_dag_in[j]]):
                    if j not in back_set and j not in visit_set:
                        new_searcher.add(j)
        searcher = deepcopy(new_searcher)
    
 
    red_scc_ins = []
    for i in red_scc:
        in_specy = set()
        for k in i:
            for j in red_in_graph[k]:
                if j not in i:
                    in_specy.add(j)
        red_scc_ins.append(in_specy)
        
    in_specy_set = set()
    for i in range(len(red_scc_ins)):
        for j in red_scc_ins[i]:
            in_specy_set.add(j)

    red_scc_order = []

    count = 0

    for i in visited:
        if i in red_scc_set:
            red_scc_order.append(red_scc_dict_2[i])
    
    red_scc_ins_2 = []
    for i in red_scc:
        insy = set()
        for j in i:
            mark = False
            for k in red_in_graph[j]:
                if k not in i:
                    mark = True
            if mark:
                insy.add(j)
        red_scc_ins_2.append(insy)
    
    
    scc_ref_data_2 = {}

    new_red_scc_order = []
    for i in red_scc_order:
        if not any([x in cat_specs for x in red_scc[i]]):
            new_red_scc_order.append(i)
    red_scc_order = deepcopy(new_red_scc_order)

    for m in range(len(red_scc_order)):
        m_para = {}
        for n in range(len(reduced_mech.reactions)):
            if any([f in reduced_mech.reactions[n].reactants for f in red_scc[red_scc_order[m]]]):
                mark = True
                
                if mark:
                    m_para[n] = reduced_mech.reactions[n].multiplier
        if settings['Print Progress'] == True:
            print('Constants to Optimize: ',m,len(m_para))

    
    scc_sets = []
    for i in scc:
        scc_sets.append(set(i))
    red_scc_sets = []
    for i in red_scc:
        red_scc_sets.append(set(i))
    for i in range(len(red_scc)):
        if red_scc_sets[i] in scc_sets:
            if i in red_scc_order:
                red_scc_order.remove(i)
    if settings['Iterations']>0:
        for m in range(len(red_scc_order)):
            
            scc_rem = red_scc_order[m]
            stop_specs = set()
            for x in range(len(red_dag_out)):
                if len(red_dag_out[x])==0:
                    stop_specs.add(x)
            for x in red_scc_order[m:]:
                for j in red_scc[x]:
                    stop_specs.add(j)
            rep_sccs = set()
            for j in red_scc[red_scc_order[m]]:
                for k in red_scc_map[j]:
                    if k in scc_dict_2:
                        rep_sccs.add(scc_dict_2[k])
            data = []
            for c in range(l_c):
                daty = [{}, {}, {}]
                for f in rep_sccs:
                    for p in scc_ref_data[f][c][0]:
    
                        if p in daty:
                            daty[0][p]+= scc_ref_data[f][c][0][p]
                        else:
                            daty[0][p]= scc_ref_data[f][c][0][p] 
                    for p in scc_ref_data[f][c][1]:
    
                        if p in daty:
                            daty[1][p]+= scc_ref_data[f][c][1][p]
                        else:
                            daty[1][p]= scc_ref_data[f][c][1][p] 
                    for p in scc_ref_data[f][c][2]:
    
                        if p in daty:
                            daty[2][p]+= scc_ref_data[f][c][2][p]
                        else:
                            daty[2][p]= scc_ref_data[f][c][2][p] 
                for sp in red_scc[red_scc_order[m]]:
                    if sp not in daty[0]:
                        daty[0][sp] = 0
                    if sp not in daty[1]:
                        daty[1][sp] = 0
                    if sp not in daty[2]:
                        daty[2][sp] = 0
                data.append(daty)
            scc_ref_data_2[red_scc_order[m]] = data
    
            all_starts = []
            for c in range(l_c):
                new_graph = []
                for p in red_graphs[c]:
                    dicy = {}
                    for j in p:
                        dicy[j] = p[j]
                    new_graph.append(dicy)
    
    
                new_in = []
                for p in red_in_graph:
                    sety = set()
                    for j in p:
                        sety.add(j)
                    new_in.append(sety)
    
                new_out = []
                for p in red_out_graph:
                    sety = set()
                    for j in p:
                        sety.add(j)
                    new_out.append(sety)
    
    
                for n in range(len(red_scc)):
                    counte = 0
                    out_specs = set()
                    all_specs = set(red_scc[n])
                    for x in red_scc[n]:
                        for j in red_out_graph[x]:
                            if j not in red_scc[n]:
                                out_specs.add(j)
                                all_specs.add(j)
                    for x in red_scc_ins_2[n]:
                        counte +=1
                        new_out[x] = all_specs
                        leny = int(np.sqrt(len(red_scc[n])))
                        data, in_cyc_data = cycle_simulator_3(red_scc[n], x, red_graphs[c],red_out_graph,red_in_graph, [leny+40,2*(leny+40)],1e-6,all_specs,out_specs)
                        cycle_out_graphs[c][x] = data
                        ful_dat = {}
                        for p in data:
                            ful_dat[p] = data[p]
                        for p in in_cyc_data:
                            ful_dat[p] = in_cyc_data[p]
                        new_graph[x] = ful_dat
    
                    for y in red_scc[m]:
                        if y not in red_scc_ins_2[red_scc_order[m]]:
    
                            for s in out_graph[y]:
                                if y in new_in[s]:
                                    new_in[s].remove(y)
                            new_in[y] = set(red_scc_ins_2[red_scc_order[m]])
                            new_out[y] = set()
                            new_graph[y] = set()
                            
                graph_mat_lil = lil_matrix((spec_len,spec_len), dtype=None)
    
                edges = []
                for i in range(len(new_out)):
                    for j in new_out[i]:
                        edges.append((i,j))
                count = 0
                for i,j in edges:
                    graph_mat_lil[i,j] = 1
            
                graph_mat = graph_mat_lil.tocsr(copy=False)
            
                scc_result = connected_components(graph_mat, directed=True, connection='strong', return_labels=True)
            
                scc_dic = {}
                for i in range(len(scc_result[1])):
                    if scc_result[1][i] in scc_dic:
                        scc_dic[scc_result[1][i]].append(i)
                    else:
                        scc_dic[scc_result[1][i]] = [i]
    
                yi = get_yields_modified(roots, stop_specs, 1e-10, new_graph, new_in, new_out, back_set,red_scc, red_scc_dict_2, red_scc_set)
               

                starters = []
                for j in red_scc_ins_2[red_scc_order[m]]:
                    start_val = 0
                    for k in red_in_graph[j]:
                        if k not in red_scc[red_scc_order[m]]:
                            if j in red_graphs[c][k] and k in yi:
                                start_val+= red_graphs[c][k][j]*yi[k]
                    
                    starters.append([j, start_val])
                all_starts.append(starters)
            data_map = []
            for j in red_scc[red_scc_order[m]]:
    
                mapper1 = [1]
                mapper2 = [0]
                mapper1.append(j)
                mapper2.append(j)
                comps = []
                
                for k in red_scc_map[j]:
                    comps.append([k,1])

                mapper1.append(comps)
                mapper2.append(comps)
                data_map.append(mapper1)
            out_specs = set()
            all_specs = set(red_scc[red_scc_order[m]])
            for x in red_scc[red_scc_order[m]]:
                for j in red_out_graph[x]:
                    if j not in red_scc[red_scc_order[m]]:
                        out_specs.add(j)
                        all_specs.add(j)
    
            for j in out_specs:
    
                mapper2 = [0]
                mapper2.append(j)
                comps = []
                denom = 0
                for k in inv_map_dict[j]:
                    denom+=avg_yield_o[k]
                for k in inv_map_dict[j]:
                    comps.append([k,avg_yield_o[k]/max(1e-20,denom)])

                mapper2.append(comps)
                data_map.append(mapper2)
            
            cycle_info = [red_scc[red_scc_order[m]], all_starts, all_specs, out_specs]
            settings_des = {}
            settings_des['learn_rate'] = 0.0002
            settings_des['steps'] = settings['Iterations']
            settings_des['grad_step'] = 1.001

            m_para_map = {}
            m_para = {}
            for n in range(len(reduced_mech.reactions)):
                if any([f in reduced_mech.reactions[n].reactants for f in red_scc[red_scc_order[m]]]):
                    mark = True

                    if mark:
                        m_para_map[n] = [n]
                        m_para[n] = reduced_mech.reactions[n].multiplier
    
            reduced_mech = grad_descent_cycles(m_para, m_para_map, reduced_mech, scc_ref_data_2[red_scc_order[m]], data_map, cycle_info, settings_des, conditions, red_specs, back_set, len(new_species), red_scc[scc_rem])
            
            red_graphs = [[{} for i in range(spec_len)] for c in range(l_c)]
            red_in_graph = [set() for i in range(spec_len)]
            red_out_graph = [set() for i in range(spec_len)]
            for nn in red_rxn_reac:
                if nn not in back_set:
                    edges = [{} for c in range(l_c)]
                    rate_sums = [0 for c in range(l_c)]
    
                    for j in red_rxn_reac[nn]:
    
                        for c in range(l_c):
                            rate_sums[c] = rate_sums[c] + reduced_mech.reactions[j].multiplier*reduced_mech.reactions[j].rate[c]
                    for j in red_rxn_reac[nn]:
                        mults = [0 for c in range(l_c)]
                        for c in range(l_c):
                            mults[c] = reduced_mech.reactions[j].multiplier*reduced_mech.reactions[j].rate[c]/max(1e-20,rate_sums[c])
                        if len(reduced_mech.reactions[j].reactants) == 1:
                            type_r = 'solo'
                        else:
                            mark = True
                            for p in reduced_mech.reactions[j].reactants:
                                if p in background_spc:
                                    type_r = p
                                    mark = False
                            if mark:
                                type_r = 'double'
                        for k in reduced_mech.reactions[j].prod_dict:
    
                       
                            for c in range(l_c):
                                if k in edges[c]:
                                    if c==0:
                                        edges[c][k] = edges[c][k] + reduced_mech.reactions[j].prod_dict[k]*mults[c]
                                        red_out_graph[nn].add(k)
                                        red_in_graph[k].add(nn)
                          
                                    else:
                                        edges[c][k] = edges[c][k] + reduced_mech.reactions[j].prod_dict[k]*mults[c]
    
    
                                else:
                                    if c==0:
                                        edges[c][k] = reduced_mech.reactions[j].prod_dict[k]*mults[c]
                                        red_out_graph[nn].add(k)
                                        red_in_graph[k].add(nn)
                                      
                                    else:
                                        edges[c][k] = reduced_mech.reactions[j].prod_dict[k]*mults[c]
                    for c in range(l_c):
                        red_graphs[c][nn] = edges[c]
    
    if settings['Print Progress'] == True:
        print('Stage 12: Getting Reduced Mechanism Yield')

    
    red_yields = []     
    for c in range(l_c):
        new_graph = []
        for p in red_graphs[c]:
            dicy = {}
            for j in p:
                dicy[j] = p[j]
            new_graph.append(dicy)


        new_in = []
        for p in red_in_graph:
            sety = set()
            for j in p:
                sety.add(j)
            new_in.append(sety)

        new_out = []
        for p in red_out_graph:
            sety = set()
            for j in p:
                sety.add(j)
            new_out.append(sety)


        for n in range(len(red_scc)):
            counte = 0
            out_specs = set()
            all_specs = set(red_scc[n])
            for x in red_scc[n]:
                for j in red_out_graph[x]:
                    if j not in red_scc[n]:
                        out_specs.add(j)
                        all_specs.add(j)
            for x in red_scc_ins_2[n]:
                counte +=1
                new_out[x] = all_specs
                leny = int(np.sqrt(len(red_scc[n])))
                data, in_cyc_data = cycle_simulator_3(red_scc[n], x, red_graphs[c],red_out_graph,red_in_graph, [leny+40,2*(leny+40)],1e-6,all_specs,out_specs)
                cycle_out_graphs[c][x] = data
                ful_dat = {}
                for p in data:
                    ful_dat[p] = data[p]

                new_graph[x] = ful_dat

            for y in red_scc[n]:
                if y not in red_scc_ins_2[n]:

                    for s in red_out_graph[y]:
                        if y in new_in[s]:
                            new_in[s].remove(y)
                    new_in[y] = set(red_scc_ins_2[n])
                    new_out[y] = set()
                    new_graph[y] = set()


        yi = get_yields_modified(roots, [], 0, new_graph, new_in, new_out, back_set,red_scc, red_scc_dict_2, red_scc_set)
        red_yields.append(yi)

    if settings['Reduce Stiffness'] == True:
        
        real_rates = []
        count = 0
        for i in reduced_mech.reactions:
            real_rates.append([count, np.mean(i.rate)*i.multiplier])
            count+=1
        sorted_rates = sorted(real_rates, key=lambda x: x[1]) 
        sorted_rates_only = [i[1] for i in sorted_rates]
        if 'Reference Rate' in settings:
            ref = settings['Reference Rate']
        else:
            ref = np.median(sorted_rates_only)
        
        differences = [math.log10(max(i[1], 1e-20))- math.log10(ref) for i in sorted_rates]
        slow_these = []
        slow_these_rxns = set()
        accounted = set()
        for i in range(len(differences)):
            if differences[i]>settings['Stiffness Threshold']:
                slow_these.append(sorted_rates[i])
                slow_these_rxns.add(sorted_rates[i][0])
        numba = 10000000000000
        for s in red_rxn_reac:
            check = [x in slow_these_rxns for x in red_rxn_reac[s]]
            if all(check):
                rates_check = [real_rates[r][1] for r in red_rxn_reac[s]]
                rate_mult = min(1,100*(10**settings['Stiffness Threshold'])*ref/max(rates_check))
                for r in red_rxn_reac[s]:
                    if r not in accounted:
                        accounted.add(r)
                        reduced_mech.reactions[r].multiplier*= rate_mult
            '''elif any(check):
                pass
                print('checking')
                rates_check = []
                for x in red_rxn_reac[s]:
                    if x not in accounted:
                        rates_check.append(real_rates[x])
                        accounted.add(x)
                sorted_check = sorted(rates_check, key=lambda x: x[1], reverse = True) 
                marks = []
                multer = []
                print(sorted_check)
                for j in range(len(sorted_check)):
                    if sorted_check[j][0] in slow_these_rxns:
                        if j>0 and sorted_check[j][1]/max(1e-20,sorted_check[j-1][1])<1/numba:
                            marks.append(j)
                            min_mult = numba*sorted_check[j][1]/max(1e-20,sorted_check[j-1][1])
                            multer.append(min_mult)
                        print('in slow', sorted_check[j][1])
                    elif j>0 and sorted_check[j][1]/max(1e-20,sorted_check[j-1][1])>1/numba:
                        print('is close', sorted_check[j][1], sorted_check[j][1]/max(1e-20,sorted_check[j-1][1]))
                    else:
                        marks.append(j)
                        min_mult = numba*sorted_check[j][1]/max(1e-20,sorted_check[j-1][1])
                        multer.append(min_mult)
                        print('max mult', min_mult)
                        break
                multer_2 = []
                for i in range(len(multer)):
                    multer_2.append(np.prod(multer[i:]))
                multer = deepcopy(multer_2)
                print('multer', marks, multer)
                count = 0
                if len(multer)>0:
                    for j in range(len(sorted_check)):
                        print(j, count)
                        if j<marks[count]:
                            reduced_mech.reactions[sorted_check[j][0]].multiplier *= multer[count]
                            print('sorty', sorted_check[j][1], multer[count])
                        elif count+1<len(marks):
                            count+=1
                            reduced_mech.reactions[sorted_check[j][0]].multiplier *= multer[count]
                            print('sorty', sorted_check[j][1], multer[count])
                        else:
                            break'''

    
    if settings['Print Progress'] == True:
        print('Stage 13: Removing Reactions')

    cat_species_set = set(cat_species_indices)
    
    avoid_these_rxns = set()
    for i in range(len(reduced_mech.reactions)):
        for j in reduced_mech.reactions[i].reactants:
            if j in red_scc_set:
                if j not in cat_species_set:
                    scc_indy = red_scc_dict_2[j]
                    for k in reduced_mech.reactions[i].prod_dict:
                        if k in red_scc[scc_indy]:
                            avoid_these_rxns.add(i)
    
    
    
    shortened_reactions = []
    reaction_type_dict = {}
    count = 0
    for i in reduced_mech.reactions:
        if settings['Keep Cycle Reactions']==True:
            if count not in avoid_these_rxns:
                if len(i.reactants) == 2:
                    reac_str = str(i.reactants[0]) + ' ' + str(i.reactants[1])
                    if reac_str in reaction_type_dict:
                        reaction_type_dict[reac_str].append(i)
                    else:
                        reaction_type_dict[reac_str] = [i]
                elif 'HV' in i.rate_string:
                    reac_str = str(i.reactants[0]) + ' ' + 'HV'
                    if reac_str in reaction_type_dict:
                        reaction_type_dict[reac_str].append(i)
                    else:
                        reaction_type_dict[reac_str] = [i]
                else:
                    reac_str = str(i.reactants[0]) + ' ' + 'OTH'
                    if reac_str in reaction_type_dict:
                        reaction_type_dict[reac_str].append(i)
                    else:
                        reaction_type_dict[reac_str] = [i]
        else:
            if len(i.reactants) == 2:
                reac_str = str(i.reactants[0]) + ' ' + str(i.reactants[1])
                if reac_str in reaction_type_dict:
                    reaction_type_dict[reac_str].append(i)
                else:
                    reaction_type_dict[reac_str] = [i]
            elif 'HV' in i.rate_string:
                reac_str = str(i.reactants[0]) + ' ' + 'HV'
                if reac_str in reaction_type_dict:
                    reaction_type_dict[reac_str].append(i)
                else:
                    reaction_type_dict[reac_str] = [i]
            else:
                reac_str = str(i.reactants[0]) + ' ' + 'OTH'
                if reac_str in reaction_type_dict:
                    reaction_type_dict[reac_str].append(i)
                else:
                    reaction_type_dict[reac_str] = [i]
        count+=1
    
    for i in reaction_type_dict:
        total_rates = []
        multi = []
        for j in reaction_type_dict[i]:
            if j not in avoid_these_rxns:
                total_rates.append(np.mean(j.rate)*j.multiplier)
                multi.append(j.multiplier)
            
        total_rate = sum(total_rates)
        avg_rate = np.mean(total_rates)
        max_reac = total_rates.index(max(total_rates))
        base_rxn = reaction_type_dict[i][max_reac]
        base_rate = np.mean(reaction_type_dict[i][max_reac].rate)
        
        mult = total_rate/max(base_rate, 1e-20)
        new_prod_dict = {}
        for j in reaction_type_dict[i]:
            factor = np.mean(j.rate)*j.multiplier/max(1e-20,total_rate)
            for k in j.prod_dict:
                if k in new_prod_dict:
                    new_prod_dict[k] += j.prod_dict[k]*factor
                else:
                    new_prod_dict[k] = j.prod_dict[k]*factor
        new_rxn = deepcopy(base_rxn)
        new_rxn.prod_dict = new_prod_dict
        new_rxn.multiplier = mult
        shortened_reactions.append(new_rxn)
        new_name_reactions =[]
    if settings['Keep Cycle Reactions']==True:
        for i in avoid_these_rxns:
            shortened_reactions.append(reduced_mech.reactions[i])

    if settings['Remove Weak Reactions']==True:
        short_rxn_reac = {}
        for i in range(len(shortened_reactions)):
            for j in shortened_reactions[i].reactants:
                if j not in back_set:
                    if j in short_rxn_reac:
                        short_rxn_reac[j].add(i)
                    else:
                        short_rxn_reac[j] = set([i])
    
        include_set = set()
        for i in short_rxn_reac:
            total_rate = 0
            for j in short_rxn_reac[i]:
                total_rate+= np.mean(shortened_reactions[j].rate)*shortened_reactions[j].multiplier
            for j in short_rxn_reac[i]:
                if np.mean(shortened_reactions[j].rate)*shortened_reactions[j].multiplier > total_rate*settings['Weak Reaction Cutoff']:
                    include_set.add(j)
        red_shortened_reactions = [shortened_reactions[i] for i in include_set]
        shortened_reactions = deepcopy(red_shortened_reactions)
    new_name_reactions = []
    if settings['Remove Reactions']== True:
        
        spec_set = set()
        for i in shortened_reactions:
        
            new_reacs = []
            for j in i.reactants:
                if species_list_names_2[j] not in background_spc_n:
                    new_reacs.append(species_list_names_2[j])
                    spec_set.add(species_list_names_2[j])
            for j in i.reactants:
                if species_list_names_2[j] in background_spc_n:
                    new_reacs.append(species_list_names_2[j])
                    spec_set.add(species_list_names_2[j])
            new_prods = {}
            for j in i.prod_dict:
                if i.prod_dict[j]>0.001:
                    new_prods[species_list_names_2[j]] = i.prod_dict[j]
                    spec_set.add(species_list_names_2[j])
            j = deepcopy(i)
            j.prod_dict = new_prods
            j.reactants = new_reacs
            if len(new_prods)>0:
                new_name_reactions.append(j)
    else:
        spec_set = set()
        for i in reduced_mech.reactions:
        
            new_reacs = []
            for j in i.reactants:
                if species_list_names_2[j] not in background_spc_n:
                    new_reacs.append(species_list_names_2[j])
                    spec_set.add(species_list_names_2[j])
            for j in i.reactants:
                if species_list_names_2[j] in background_spc_n:
                    new_reacs.append(species_list_names_2[j])
                    spec_set.add(species_list_names_2[j])
            new_prods = {}
            for j in i.prod_dict:
                if i.prod_dict[j]>0.001:
                    new_prods[species_list_names_2[j]] = i.prod_dict[j]
                    spec_set.add(species_list_names_2[j])
            j = deepcopy(i)
            j.prod_dict = new_prods
            j.reactants = new_reacs
            if len(new_prods)>0:
                new_name_reactions.append(j)
 
    reduced_mech.reactions = new_name_reactions

    
    outputs = {}
    outputs['Reduced Mechanism'] = reduced_mech
    outputs['Remaining Species'] = [species_list_names_2[i] for i in remaining_specs]
    outputs['Removed Species'] = [species_list_names_2[i] for i in remove_species]
    outputs['Removal Order'] = [species_list_names_2[i] for i in ordered_species]
    
    outputs['Full Graphs'] = []
    for i in graphs:
        new_graph = {}
        for j in range(len(i)):
            new_edges = {}
            for k in i[j]:
                new_edges[species_list_names_2[k]] = i[j][k]
            new_graph[species_list_names_2[j]] = new_edges
        outputs['Full Graphs'].append(new_graph)
    
    outputs['Full Yields'] = []
    for i in yields:
        new_yield = {}
        for j in i:
            new_yield[species_list_names_2[j]] = i[j]
        outputs['Full Yields'].append(new_yield)
    
    outputs['Average Full Yield'] = {}
    for i in avg_yield:
        outputs['Average Full Yield'][species_list_names_2[i]] = avg_yield[i]
        
    outputs['Reduced Graphs'] = []
    for i in red_graphs:
        new_graph = {}
        for j in range(len(i)):
            new_edges = {}
            for k in i[j]:
                new_edges[species_list_names_2[k]] = i[j][k]
            new_graph[species_list_names_2[j]] = new_edges
        outputs['Reduced Graphs'].append(new_graph)
    
    outputs['Reduced Yields'] = []
    for i in red_yields:
        new_yield = {}
        for j in i:
            new_yield[species_list_names_2[j]] = i[j]
        outputs['Reduced Yields'].append(new_yield)
    
    outputs['Tiers'] = []
    for i in tiers:
        new = set()
        for j in i:
            new.add(species_list_names_2[j])
        outputs['Tiers'].append(new)
        
    outputs['Strongly Connected Components'] = []
    for i in scc:
        new_scc = []
        for j in i:
            new_scc.append(species_list_names_2[j])
        outputs['Strongly Connected Components'].append(new_scc)
    
    outputs['Reduced Strongly Connected Components'] = []
    for i in red_scc:
        new_scc = []
        print(i)
        for j in i:
            new_scc.append(species_list_names_2[j])
        outputs['Reduced Strongly Connected Components'].append(new_scc)
    
    outputs['Groups'] = []
    for i in groups:
        new_group = []
        for j in i:
            new_group.append(species_list_names_2[j])
        outputs['Groups'].append(new_group)
    
    outputs['Species Dictionary'] = dic
    outputs['Species'] = species_list_names_2
    
    print('Reduction Complete!')
    return outputs
    #return reduced_mech, categories, test_mechanism, remaining_specs, avg_yield, yields, graphs, red_yields, red_graphs, new_graph, scc, red_scc, groups, species_list, dic, tiers, dag_graph_save, dag_graph_save_2, inv_map_dict, mult_list_list, cat_data, multy, ordered_species, avg_yield_o, avg_yield_o2
    

    
                

class Mechanism:
    def __init__(self, species, reactions):
        self.species = species
        self.reactions = reactions

class Reaction:
    def __init__(self, reactants, prod_dict, rate_law, eval_rate_law, rate, rate_string = '', multiplier = 1):
        self.reactants = reactants
        self.prod_dict = prod_dict
        self.rate_law = rate_law
        self.eval_rate_law = eval_rate_law
        self.rate = rate
        self.rate_string = rate_string
        self.multiplier = multiplier


def round_it(x, sig):
    if abs(x)>0:
        return round(x, sig-int(math.floor(math.log10(abs(x))))-1)
    else:
        return 0


def cycle_simulator_3(cycle_species, in_spec, graph, out_graph, in_graph, iteration_set, cutoff, all_spec, out_spec):
    data_set = []
    search = {in_spec:1}
    counter = 0
    data = {j:0 for j in all_spec}
    out_dat = {j:0 for j in cycle_species}
    while counter<iteration_set[1]:
        new_search = {}
        if counter == iteration_set[0]:
            data_1 = {p:data[p] for p in data}
            search_1 = {p:search[p] for p in search}
        for k in search:
            #data[k] = data[k]-search[k]
            return_amount = 0
            for x in graph[k]:
                data[x] = data[x] + graph[k][x]*search[k]
                
                if x in cycle_species:
                    return_amount = return_amount + search[k]*graph[k][x]
                    if graph[k][x]*search[k]>cutoff:
                        if x in new_search:
                            new_search[x] = new_search[x] + graph[k][x]*search[k]
                        else:
                            new_search[x] = graph[k][x]*search[k]
            out_dat[k] = out_dat[k] + search[k]-return_amount
        search = {p:new_search[p] for p in new_search}
        counter = counter + 1
    frac1 = 0
    frac2 = 0
    for n in cycle_species:
        if n in search_1:
            frac1 = frac1 + search_1[n]
        if n in search:
            frac2 = frac2 + search[n]
    for i in out_dat:
        out_dat[i] = max(0,out_dat[i]/max(1-frac2,1e-20))
    final_data = {}
    within_data = {}
    denom = frac2-frac1
    if denom ==0:
        denom = 1e-20
    for n in out_spec:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        final_data[n] = dat
    for n in cycle_species:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        within_data[n] = dat
    within_data_2 = {}
    within_data_sum = 0
    for i in within_data:
        within_data_sum = within_data_sum + within_data[i]
    for i in within_data:
        within_data_2[i] = abs(within_data[i]/max(within_data_sum,1e-20))

    final_in_data = {}
    for i in within_data_2:
        final_in_data[i] = np.mean([within_data_2[i],out_dat[i]])
    return final_data, final_in_data #, data



def cycle_simulator_3_out(cycle_species, in_spec, graph, out_graph, in_graph, iteration_set, cutoff, all_spec, out_spec):
    data_set = []
    search = {in_spec:1}
    counter = 0
    data = {j:0 for j in all_spec}
    out_dat = {j:0 for j in cycle_species}
    while counter<iteration_set[1]:
        new_search = {}
        if counter == iteration_set[0]:
            data_1 = {p:data[p] for p in data}
            search_1 = {p:search[p] for p in search}
        for k in search:
            #data[k] = data[k]-search[k]
            return_amount = 0
            for x in graph[k]:
                data[x] = data[x] + graph[k][x]*search[k]
                
                if x in cycle_species:
                    return_amount = return_amount + search[k]*graph[k][x]
                    if graph[k][x]*search[k]>cutoff:
                        if x in new_search:
                            new_search[x] = new_search[x] + graph[k][x]*search[k]
                        else:
                            new_search[x] = graph[k][x]*search[k]
            out_dat[k] = out_dat[k] + search[k]-return_amount
        search = {p:new_search[p] for p in new_search}
        counter = counter + 1
    frac1 = 0
    frac2 = 0
    for n in cycle_species:
        if n in search_1:
            frac1 = frac1 + search_1[n]
        if n in search:
            frac2 = frac2 + search[n]
    for i in out_dat:
        out_dat[i] = max(0,out_dat[i]/max(1-frac2,1e-20))
    final_data = {}
    within_data = {}
    denom = frac2-frac1
    if denom ==0:
        denom = 1e-20
    for n in out_spec:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        final_data[n] = dat
    for n in cycle_species:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        within_data[n] = dat
    within_data_2 = {}
    within_data_sum = 0
    for i in within_data:
        within_data_sum = within_data_sum + within_data[i]
    for i in within_data:
        within_data_2[i] = abs(within_data[i]/max(within_data_sum,1e-20))

    final_in_data = {}
    for i in within_data_2:
        final_in_data[i] = np.mean([within_data_2[i],out_dat[i]])
    return final_data, final_in_data, out_dat #, data



def cycle_simulator_3_for_test(cycle_species, in_spec, graph, out_graph, in_graph, iteration_set, cutoff, all_spec, out_spec):
    data_set = []
    search = {in_spec:1}
    counter = 0
    data = {j:0 for j in all_spec}
    out_dat = {j:0 for j in cycle_species}
    while counter<iteration_set[1]:
        new_search = {}
        if counter == iteration_set[0]:
            data_1 = {p:data[p] for p in data}
            search_1 = {p:search[p] for p in search}
        for k in search:
            #data[k] = data[k]-search[k]
            return_amount = 0
            for x in graph[k]:
                if x in data:
                    data[x] = data[x] + graph[k][x]*search[k]
                #else:
                 #   data[x] = graph[k][x]*search[k]
                
                if x in cycle_species:
                    return_amount = return_amount + search[k]*graph[k][x]
                    if graph[k][x]*search[k]>cutoff:
                        if x in new_search:
                            new_search[x] = new_search[x] + graph[k][x]*search[k]
                        else:
                            new_search[x] = graph[k][x]*search[k]
            if k in out_dat:
                out_dat[k] = out_dat[k] + search[k]-return_amount
           # else:
            #    out_dat[k] = search[k]-return_amount
        search = {p:new_search[p] for p in new_search}
        counter = counter + 1
    frac1 = 0
    frac2 = 0
    for n in cycle_species:
        if n in search_1:
            frac1 = frac1 + search_1[n]
        if n in search:
            frac2 = frac2 + search[n]
    for i in out_dat:
        out_dat[i] = max(0,out_dat[i]/max(1-frac2,1e-20))
    final_data = {}
    within_data = {}
    denom = frac2-frac1
    if denom ==0:
        denom = 1e-20
    for n in out_spec:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        final_data[n] = dat
    for n in cycle_species:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        within_data[n] = dat
    within_data_2 = {}
    within_data_sum = 0
    for i in within_data:
        within_data_sum = within_data_sum + within_data[i]
    for i in within_data:
        within_data_2[i] = abs(within_data[i]/max(within_data_sum,1e-20))

    final_in_data = {}
    for i in within_data_2:
        final_in_data[i] = np.mean([within_data_2[i],out_dat[i]])
    return final_data, final_in_data#, data



def copy_mechanism(mechanism):
    species_list = []
    reactions = []
    for i in range(len(mechanism.species)):
        species_list.append(deepcopy(mechanism.species[i]))
    for i in range(len(mechanism.reactions)):
        reactions.append(deepcopy(mechanism.reactions[i]))
    new_mechanism = Mechanism(species_list, reactions)
    return new_mechanism



#KPP eqns to python (rxn,rate) format
#as an input and outputs the reactions as a list in the format of [[reaction, rate],[r2],[r3]...]
def read_eqns(eqn_file):
    '''Read .eqn files
    Parameters
    ----------
    eqn_file: .eqn file
      The .eqn file to read

    Returns
    ----------
    species: list
      A list of tuples. The first element in the tuple is an equation.
      The second element in the tuple is reaction rate. 
    '''
  
    equations = None
    with open(eqn_file,'r') as f:
        lines = f.readlines()
        equantions = lines[:]
    equations = [i.strip() for i in equantions[1:]]
    equations = [tuple(i.split(':')) for i in equations if len(i)>0]
    equations = [[i[0].strip(),i[1].split(';')[0].strip()] for i in equations if len(i)>1]
    return(equations)



#KPP spc to python 
#as an input and outputs the reactions as a list in the format of [species1,species2,...]
def read_spc(spc_file):
    '''Read .spc files and process the raw input into species
    Parameters
    ----------
    spc_file: .eqn file
        The .spc file to read
    
    Returns
    ----------
    species: list
        List of species.
    '''
    species = None
    with open(spc_file,'r') as f:
        lines = f.readlines()
        species = lines[:]
    species = [s.split('=')[0].strip() for s in species]
    species = [s for s in species if s and s[0]!='#']
    return(species)



#python functions for all of the rate constant functions given in the mechanism file. 
#These were copied from isoprene_rates.py, made by the DSI team
def ISO1(TEMP, A0, B0, C0, D0, E0, F0, G0):
    K0 = D0 * EXP(E0/TEMP) * EXP(1.E8/TEMP**3)
    K1 = F0 * EXP(G0/TEMP)
    K2 = C0 * K0/(K0+K1)
    ISO1 = A0 * EXP(B0/TEMP) * (-(K2-1))
    return ISO1
def EXP(x):
    return(math.exp(x))
def LOG10(x):
    return(math.log10(x))
def TUN(TEMP, A0, B0, C0):
    return(A0 * EXP(-B0/TEMP) * EXP(C0/TEMP**3))
def ALK(TEMP, M, A0, B0, C0, n, X0, Y0):
    K0 = 2.0E-22 * EXP(n)
    K1 = 4.3E-1 * (TEMP/298.0) ** (-8)
    K0 = K0 * M
    K1 = K0/K1
    K2 = (K0/(1.0 + K1)) * 4.1E-1 ** (1.0/(1.0 + (LOG10(K1)) ** 2))
    K3 = C0/(K2 + C0)
    K4 = A0 * (X0 - TEMP * Y0)
    ALK = K4 * EXP(B0/TEMP) * K3
    return(ALK)
def NIT(TEMP, M, A0, B0, C0, n, X0, Y0):
    K0 = 2.0E-22 * EXP(n)
    K1 = 4.3E-1 *(TEMP/298.0) ** (-8)
    K0 = K0 * M
    K1 = K0/K1
    K2 = (K0/(1.0 + K1)) * 4.1E-1 ** (1.0 /(1.0 +(LOG10(K1)) ** 2))
    K3 = K2/(K2 + C0)
    K4 = A0*(X0 - TEMP * Y0)
    NIT = K4 * EXP(B0/TEMP) * K3
    return(NIT)
def ISO2(TEMP, A0, B0, C0, D0, E0, F0, G0):
    K0 = D0 * EXP(E0/TEMP) * EXP(1.E8/TEMP**3)
    K1 = F0 * EXP(G0/TEMP)
    K2 = C0 * K0/(K0+K1)
    ISO2 = A0 * EXP(B0/TEMP) * K2
    return ISO2
def EPO(TEMP, A1, E1, M1):
    K1 = 1 / (M1 * 2.5e+19 + 1)
    #K1 = 1 / (M1 * CFACTOR + 1)
    EPO = A1 * EXP(E1/TEMP) * K1
    return EPO
def KCO(A1, M1):
    KCO = A1 * (1 + (2.5e+19/M1))
    #KCO = A1 * (1 + (CFACTOR/M1))
    return KCO
def FALL(TEMP, A0,B0,C0,A1,B1,C1,CF):
    K0 = A0 * EXP(B0/TEMP) * (TEMP/300)**C0
    K1 = A1 * EXP(B1/TEMP) * (TEMP/300)**(C1)
    K0 = K0*2.5e+19
    #K0 = K0*CFACTOR
    K1 = K0/K1
    FALL = (K0 / (1.00+K1) * CF**(1 / (1 + LOG10(K1)) **2))
    return FALL
def TROE(TEMP, M, A0, B0, C0, A1, B1, C1, CF):
    K0 = A0 * EXP(B0/TEMP) * (TEMP/300) ** C0
    K1 = A1 * EXP(B1/TEMP) * (TEMP/300) ** C1
    K0 = K0 * M
    KR = K0/K1
    NC = 0.75 - 1.27 * LOG10(CF)
    F = 10 ** ((LOG10(CF)) / (1+(LOG10(KR)/NC)**2))
    TROE = K0*K1*F / (K0+K1)
    return TROE
def ARR(TEMP, A0, B0, C0):
    return(A0 * EXP(B0/TEMP) * EXP(C0/TEMP**3))
def K_OH_CO(T,M):
    T3I = 1/T
    KLO1=5.9e-33*(300*T3I)**(1.4)
    KHI1=1.1E-12*(300.*T3I)**(-1.3)
    XYRAT1=(KLO1*M)/KHI1
    BLOG1= np.log10(XYRAT1)
    FEXP1=1.0/(1.0+BLOG1*BLOG1)
    KCO1=KLO1*M*0.6**(FEXP1)/(1.0+XYRAT1)
    KLO2=1.5E-13*(300*T3I)**(-0.6)
    KHI2=2.1E9*(300*T3I)**(-6.1)
    XYRAT2 = KLO2*M/KHI2
    BLOG2=LOG10(XYRAT2)
    FEXP2=1.0/(1.0+ BLOG2*BLOG2)
    KCO2=KLO2*0.6**(FEXP2/(1.0+XYRAT2))
    KCO=KCO1+KCO2
    return KCO
def KRO2NO3():
    return 2.3e-12
def KAPHO2(T):
    return 5.2e-13*np.exp(980/T) 
def KAPNO(T):
    return 7.5e-12*np.exp(290/T)
def KNO3AL(T):
    return 1.44e-12*np.exp(-1862/T) ;
def KCH3O2(T):
    return 1.03e-13*np.exp(365/T)
def KBPAN(T,M):
    KD0 = 1.10e-05*M*np.exp(-10100/T)
    KDI = 1.90e17*np.exp(-14100/T)
    KRD = KD0/KDI
    FCD = 0.30
    NCD = 0.75-1.27*(np.log10(FCD)) ;
    FD = 10**(np.log10(FCD)/(1+(np.log10(KRD)/NCD)**2)) ;
    return (KD0*KDI)*FD/(KD0+KDI) ;
def KFPAN(T,M):
    KC0 = 3.28e-28*M*(T/300)**(-6.87) 
    KCI = 1.125e-11*(T/300)**(-1.105) 
    KRC = KC0/KCI 
    FCC = 0.30 
    NC = 0.75-1.27*(np.log10(FCC)) 
    FC = 10**(np.log10(FCC)/(1+(np.log10(KRC)/NC)**2)) ;
    return (KC0*KCI)*FC/(KC0+KCI) ;



def eq_string(rxn):
    eq_string = ''
    for i in range(len(rxn[0])):
        if i < len(rxn[0])-1:
            eq_string = eq_string + str(rxn[1][i]) + rxn[0][i] + ' + '
        else:
            eq_string = eq_string + str(rxn[1][i]) + rxn[0][i] + ' = '
    for j in range(len(rxn[3])):
        if i < len(rxn[3])-1:
            eq_string = eq_string + str(rxn[4][j]) + rxn[3][j] + ' + '
        else:
            eq_string = eq_string + str(rxn[4][j]) + rxn[3][j]
    return eq_string


def get_n_balance(reaction):
    n = 0
    for i in reaction[0]:
        ind = species_list.index(i[0])
        n = n + species_list_n[ind]*i[1]
    return n


def balance_n(reaction):
    x = get_n_balance(reaction)
    n_list = []
    for i in reaction[0]:
        if species_list_n[species_list.index(i[0])]>0:
            if i[1]>0:
                n_list.append(species_list_n[species_list.index(i[0])]*i[1])
    if sum(n_list)!=0:
        multiplier = -x/sum(n_list)
    else:
        multiplier = 0
    new_reaction = [[],reaction[1]]
    for i in reaction[0]:
        if species_list_n[species_list.index(i[0])]==0:
            new_reaction[0].append(i)
        elif i[1] < 0:
            new_reaction[0].append(i)
        else:
            if species_list_n[species_list.index(i[0])]>0:
                new_reaction[0].append([i[0],i[1]+i[1]*multiplier])
    return new_reaction



#pressure_to_m(P,T)
#number density is molecules per cubic centimeter
#it is n = p/kT, if p is in mbar, then that is equivalent to 100 pascals, so we multiple k by 
def pressure_to_m(P,T):
    Na = 6.022e23; #molecules per mole
    R = 8.314e4; # cm^3 mbar /K /mol
    M = Na*P/(R*T)
    return M


def gen_analysis(species, in_graph, graph, generations, yields_of):
    count = 0
    current_list = set([species])
    while count<generations:
        new_list = set()
        for i in current_list:
            for j in in_graph[i]:
                new_list.add(j)
        current_list = deepcopy(new_list)
        count+=1
    yields = []
    for i in current_list:

        if i in yields_of:
            yields.append([i,yields_of[i]])
    return yields


def solve_j_rate(i, sza,SUN):
    if i == 'J(22)' or i== 'J22':
        I = 5.804e-6;
        m = 1.092;
        n = 0.377;
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(34)' or i== 'J34':
        I = 1.537e-4;
        m = 0.170;
        n = 0.208;
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(41)' or i== 'J41':
        I = 7.649e-6;
        m = 0.682;
        n = 0.279;
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(31)' or i== 'J31':
        I = 6.845e-5
        m = 0.130
        n = 0.201
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(32)' or i== 'J32':
        I = 1.032e-5
        m = 0.130
        n = 0.201
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(33)' or i== 'J33':
        I = 3.802e-5
        m = 0.644
        n = 0.312
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(11)' or i== 'J11':
        I = 4.642e-5
        m = 0.762
        n = 0.353
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(12)' or i== 'J12':
        I = 6.853e-5
        m = 0.477
        n = 0.323
        k = j_func(sza,I,m,n)*SUN 
    elif i == 'J(15)' or i== 'J15':
        I = 2.792e-5
        m = 0.805
        n = 0.338
        k = j_func(sza,I,m,n)*SUN
    elif i == 'J(51)' or i== 'J51':
        I = 1.588e-6;
        m = 1.154;
        n = 0.318;
        k = j_func(sza,I,m,n)*SUN
    return k



def get_k_list(rate_list, sza):
    k_list = []
    for i in rate_list:
        if i == 'J(22)' or i== 'J22':
            I = 5.804e-6;
            m = 1.092;
            n = 0.377;
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(34)' or i== 'J34':
            I = 1.537e-4;
            m = 0.170;
            n = 0.208;
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(41)' or i== 'J41':
            I = 7.649e-6;
            m = 0.682;
            n = 0.279;
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(31)' or i== 'J31':
            I = 6.845e-5
            m = 0.130
            n = 0.201
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(32)' or i== 'J32':
            I = 1.032e-5
            m = 0.130
            n = 0.201
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(33)' or i== 'J33':
            I = 3.802e-5
            m = 0.644
            n = 0.312
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(11)' or i== 'J11':
            I = 4.642e-5
            m = 0.762
            n = 0.353
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(12)' or i== 'J12':
            I = 6.853e-5
            m = 0.477
            n = 0.323
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(15)' or i== 'J15':
            I = 2.792e-5
            m = 0.805
            n = 0.338
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif i == 'J(51)' or i== 'J51':
            I = 1.588e-6;
            m = 1.154;
            n = 0.318;
            k_list.append(j_func(sza,I,m,n)*SUN)
        elif isinstance(i,str):
            k_list.append(eval(i))
        else:
            k_list.append(i) 
    return k_list


#get_prod_reac(eq_list)
#takes eq list and separates into format reac_list2 = [[r1,r2],[r1],...], prod_list2 = [[r1,r2],[r1],...], 
#reac_coeff_list = [[1,1],[1],...], prod_coeff_list = [[1,1],[2],...] (as examples)
def get_prod_reac(eq_list):
    nums = ['0','1','2','3','4','5','6','7','8','9']
    reac_list = [i[0].split(' = ')[0].split(' + ') for i in eq_list]
    prod_list = [i[0].split(' = ')[1].split(' + ') for i in eq_list]
    prod_list2 = deepcopy(prod_list)
    prod_coeff_list = deepcopy(prod_list)
    reac_list2 = deepcopy(reac_list)
    reac_coeff_list = deepcopy(reac_list)
    a=[]
    for i in range(len(prod_list2)):
        for j in range(len(prod_list2[i])):
            if prod_list2[i][j][:1] in nums:
                a=re.split('([a-zA-Z])',prod_list[i][j],1)
                prod_list2[i][j]=a[1]+a[2]
                prod_coeff_list[i][j]=a[0]
            else:
                prod_coeff_list[i][j]=1
    b=[]
    for i in range(len(reac_list2)):
        for j in range(len(reac_list2[i])):
            if reac_list2[i][j][:1] in nums:
                b=re.split('([a-zA-Z])',reac_list[i][j],1)
                reac_list2[i][j]=b[1]+b[2]
                reac_coeff_list[i][j]=b[0]
            else:
                reac_coeff_list[i][j]=1
    for i in range(len(prod_coeff_list)):
        for j in range(len(prod_coeff_list[i])):
            prod_coeff_list[i][j] = float(prod_coeff_list[i][j])
    for i in range(len(reac_coeff_list)):
        for j in range(len(reac_coeff_list[i])):
            reac_coeff_list[i][j] = float(reac_coeff_list[i][j])
    return reac_list2,reac_coeff_list,prod_list2,prod_coeff_list


#is_reachable(species_A,species_B,out_list)
#determines weather B is reachable from A
def is_reachable(species_A,species_B,out_list):
    test_list=[species_A]
    test_list_up = test_list
    visited_list = []
    while len(test_list)>0:
        for i in test_list:
            visited_list.append(i)
            for j in out_list[i]:
                if j not in visited_list:
                    test_list_up.append(j)
            test_list_up.remove(i)
        test_list = test_list_up
    if species_B in visited_list:
        return True
    else:
        return False


#j_func(sza,I,m,n)
#J = I * cos(SZA)^m * exp(-n * sec(SZA))
def j_func(sza,I,m,n):
    a = I*(np.cos(sza*(2*np.pi)/360)**m)
    b = np.exp(-n/(np.cos(sza*(2*np.pi)/360)))
    return a*b


def rxn_index_convert(reac,prod,background_spc_n, background_spc, reac_len,spec_len, dic):
    reac_list = []
    prod_list = []
    background_spc_n_dic = {}
    for i in range(len(background_spc_n)):
        background_spc_n_dic[background_spc_n[i]] = i
    for i in range(reac_len):
        rx = []
        for j in reac[i]:
            rx.append(j)
        reac_list.append(rx)
        pd = []
        for j in prod[i]:
            pd.append(j)
        prod_list.append(pd)
    
    reac_no_back = [[] for i in range(reac_len)]
    prod_no_back = [[] for i in range(reac_len)]
    rxn_reac = [[] for i in range(spec_len)]
    rxn_prod = [[] for i in range(spec_len)]
    for i in range(reac_len):
        #if i%10000 == 0:
         #   print(i)
        for j in range(len(reac_list[i])):
            spec = reac[i][j]
            if spec in background_spc_n:
                ind = background_spc[background_spc_n_dic[spec]]
                reac_list[i][j] = ind
                rxn_reac[ind].append(i)
            else:
                ind = dic.get(spec)
                rxn_reac[ind].append(i)
                reac_list[i][j] = ind
                reac_no_back[i].append(ind)
        for k in range(len(prod_list[i])):
            spec = prod[i][k]
            if spec in background_spc_n:
                ind = background_spc[background_spc_n_dic[spec]]
                prod_list[i][k] = ind
                rxn_prod[ind].append(i)
            else:
                ind = dic.get(spec)
                prod_list[i][k] = ind
                prod_no_back[i].append(ind)
                
                rxn_prod[ind].append(i)
    return reac_list, prod_list, reac_no_back, prod_no_back, rxn_reac, rxn_prod 


def copy_graph(graph, in_graph, out_graph):
    new_graph = []
    for i in graph:
        dicy = {}
        for j in i:
            dicy[j] = i[j]
        new_graph.append(dicy)


    new_in = []
    for i in in_graph:
        sety = set()
        for j in i:
            sety.add(j)
        new_in.append(sety)

    new_out = []
    for i in out_graph:
        sety = set()
        for j in i:
            sety.add(j)
        new_out.append(sety)
    return graph, in_graph, out_graph


def get_yields_modified(starts, ends, cutoff, graph, in_graph, out_graph, back_set,scc,scc_dic, scc_set):
    yields = {}
    search_list = {}
    for i in starts:
        yields[i] = 1
        search_list[i] = 1
    while len(search_list)>0:
        new_list = {}
        for i in search_list:
           # if i==0:
           #     print(graph[i])
            for j in graph[i]:

                val = search_list[i]*graph[i][j]
                if j in yields:
                    yields[j] = yields[j] + val
                else:
                    yields[j] = val
                truth = False
                if j in scc_set and i in scc_set:
                    if scc_dic[j]==scc_dic[i]:
                        truth = True

                if val>cutoff and j not in back_set and not truth and j not in ends:
                    if j in new_list:
                        new_list[j] = new_list[j] + val
                    else:
                        new_list[j] = val
                   # print(j, graph[j])
        search_list = {i:new_list[i] for i in new_list}
        #print(random.choice(list(search_list.keys())))
    new_yields = {}
    for i in yields:
        if i in ends:
            new_yields[i] = yields[i]
    return yields                 


def get_yields(starts, cutoff, graph, in_graph, out_graph, back_set,scc,scc_dic, scc_set, slen):
    yields = {i:0 for i in range(slen)}
    search_list = {}
    for i in starts:
        yields[i] = 1
        search_list[i] = 1
    while len(search_list)>0:
        #print(len(search_list))
        new_list = {}
        for i in search_list:
            for j in graph[i]:
                val = search_list[i]*graph[i][j]
                yields[j] = yields[j] + val
                truth = False
                if j in scc_set and i in scc_set:
                    if scc_dic[j]==scc_dic[i]:
                        truth = True

                if val>cutoff and j not in back_set and not truth:
                    if j in new_list:
                        new_list[j] = new_list[j] + val
                    else:
                        new_list[j] = val
        search_list = {i:new_list[i] for i in new_list}
    return yields                 



class Reaction:
    def __init__(self, reactants, prod_dict, rate_law, eval_rate_law, rate, rate_string = '', multiplier = 1):
        self.reactants = reactants
        # list of reactants in the reaction
        self.prod_dict = prod_dict
        # dictionary with product names as keys and stoichiometric coefficients as values
        self.rate_law = rate_law
        # the rate law written as an evaluatable statement (any named function must be defined)
        # optional, if eval_rate_law is used instead, put 'null' here.
        self.eval_rate_law = eval_rate_law
        # the evaluated rate law value, must provided as list containing a value for each atmospheric condition
        # if rate_law = 'null', these values will be used, otherwise rate_law will be used to calculate the rate constants and list them here
        self.rate = rate
        # This is the rate, which does not need to be input, but is used in the algorithm to save a list of the relative rate (each item is for an atmospheric condition)
        self.rate_string = rate_string
        # For any specific formatting (such as GECKO or KPP), it is useful to have the original rate law string, which is then modified rather than replaced
        # Any relevant strings for mechanism formatting can go here, they won't be changed
        self.multiplier = multiplier
        # This is the mechanism multiplying factor, which will change throughout the algorithm but should be input as 1
        # No input is required as it defaults to 1


def get_yields_from_mech(mechanism, background_spc_n, conditions):
    #creating necessary variables/objects
    protected = set(settings['Protected']).union(background_spc_n)
    print('1')
    needed_concentrations = set()
    print('1.0')
    test_reactions = []
    for i in range(len(mechanism.reactions)):
        #if i not in settings['Background Rxns'] and i not in settings['Aerosol Rxns']:
        test_reactions.append(deepcopy(mechanism.reactions[i]))
    test_mechanism = Mechanism(deepcopy(mechanism.species),test_reactions)
    
    print('1.1')
    reac_len = len(test_mechanism.reactions)

    spec_len = len(test_mechanism.species)
    dic = {test_mechanism.species[i]:i for i in range(spec_len)}
    species_list = [i for i in range(spec_len)]
    roots = [dic[i] for i in settings['roots']]
    l_c = len(conditions)
    print(roots)
    print('1.2')
    back_set = set()
    for i in background_spc_n:
        back_set.add(mechanism.species.index(i))
    prod_coeff_list = []
    prod_dict = [{} for i in range(reac_len)]
    print('1.3')
    reac_list_n = []
    prod_list_n = []
    reac_len_new = deepcopy(reac_len)
    for i in range(reac_len):
        test_mechanism.reactions[i].rate = []
        reac_list_n.append(test_mechanism.reactions[i].reactants)
        prod_list_n.append(list(test_mechanism.reactions[i].prod_dict))
        for k in test_mechanism.reactions[i].prod_dict:
            prod_dict[i][dic[k]] = test_mechanism.reactions[i].prod_dict[k]

    print('1.4')
    reac_list, prod_list, reac_no_back, prod_no_back, rxn_reac, rxn_prod = rxn_index_convert(reac_list_n,prod_list_n,background_spc_n, background_spc, reac_len, spec_len, dic)
    print('1.5')
    
    rxn_prod = [set(i) for i in rxn_prod]
    rxn_reac = [set(i) for i in rxn_reac]
    print('2')
    #measuring rates for all conditions
    county = 0
    for i in conditions: 
        M = pressure_to_m(i['pressure'],i['temp'])
        TEMP = i['temp']
        CFACTOR = 2.5e+19
        SUN = i['sun']
        p_fac = M/1000000000
        double_rxns = []
        count = 0
        for j in test_mechanism.reactions:
            if j.rate_law!='null':
                if 'J' in j.rate_law:
                    
                    j.eval_rate_law = solve_j_rate(j.rate_law, i['sza'],i['sun'])

                else:
                    j.eval_rate_law = eval(j.rate_law)
                if len(j.reactants)==1:
                    j.rate.append(j.eval_rate_law)
                elif j.reactants[1] in background_spc_n and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in background_spc_n and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[0]]*p_fac)
                elif j.reactants[1] in i and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in i and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[0]]*p_fac)
                else:
                    needed_concentrations.add(j.reactants[1])
                    root_conc = 0
                    for r in settings['roots']:
                        root_conc = i[r] + root_conc
                    j.rate.append(j.eval_rate_law*0.001*root_conc*p_fac)
                    double_rxns.append(count)
                count = count + 1
                print(j.reactants, j.prod_dict, j.rate_law, j.eval_rate_law)
            else:
                if len(j.reactants)==1:
                    j.rate.append(j.eval_rate_law[county])
                elif j.reactants[1] in background_spc_n and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[county]*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in background_spc_n and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[county]*i[j.reactants[0]]*p_fac)
                elif j.reactants[1] in i and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[county]*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in i and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[county]*i[j.reactants[0]]*p_fac)
                else:
                    needed_concentrations.add(j.reactants[1])
                    root_conc = 0
                    for r in settings['roots']:
                        root_conc = i[r] + root_conc
                    j.rate.append(j.eval_rate_law[county]*0.001*root_conc*p_fac)
                    double_rxns.append(count)
        county+=1
            #print(j.reactants,j.eval_rate_law)
    print('For better results, add concentrations for these species:', needed_concentrations)
    print('3')
    # identifying weak reactions which can be removed
    '''if settings['Remove weak reactions'] == True:
        keep_rxns = set()
        for i in range(spec_len):
            rate_list = []
            rate_dict = {}
            for j in rxn_reac[i]:
                dat = max(test_mechanism.reactions[j].rate)
                rate_list.append(dat)
                rate_dict[j] = dat
            if len(rate_list)>0:
                max_r = max(rate_list)
            else:
                max_r = 0
            for k in rate_dict:
                if rate_dict[k]>max_r*settings['Weak rxn cutoff']:
                    keep_rxns.add(k)
                #else:
                    #print(k, rate_dict[k], max_r, test_mechanism.reactions[k].rate)
        remove_rxns = set(range(reac_len)) - keep_rxns'''
        
    # how many species can we get rid of from the weak rxns?

    
        
    print('6')
    graphs = [[] for c in range(l_c)]
    in_graph = [set() for i in range(spec_len)]
    out_graph = [set() for i in range(spec_len)]
    out_graph_type = [{} for i in range(spec_len)]
    in_graph_type  = [{} for i in range(spec_len)]
    
    
    
    
    
    for i in range(spec_len):
        if i not in back_set:
            edges = [{} for c in range(l_c)]
            rate_sums = [0 for c in range(l_c)]
            
            for j in rxn_reac[i]:
                for c in range(l_c):
                    rate_sums[c] = rate_sums[c] + test_mechanism.reactions[j].rate[c]
            for j in rxn_reac[i]:
                mults = [0 for c in range(l_c)]
                for c in range(l_c):
                    mults[c] = test_mechanism.reactions[j].rate[c]/max(1e-20,rate_sums[c])
                if len(test_mechanism.reactions[j].reactants) == 1:
                    type_r = 'solo'
                else:
                    mark = True
                    for p in reac_list[j]:
                        if p in background_spc:
                            type_r = p
                            mark = False
                    if mark:
                        type_r = 'double'
                for k in prod_dict[j]:
                    if i in in_graph_type[k]:
                        in_graph_type[k][i].add(type_r)
                    else:
                        in_graph_type[k][i] = set([type_r])
                    for c in range(l_c):
                        if k in edges[c]:
                            if c==0:
                                edges[c][k] = edges[c][k] + prod_dict[j][k]*mults[c]
                                out_graph[i].add(k)
                                in_graph[k].add(i)
                                out_graph_type[i][k].add(type_r)
                            else:
                                edges[c][k] = edges[c][k] + prod_dict[j][k]*mults[c]

                        
                        else:
                            if c==0:
                                edges[c][k] = prod_dict[j][k]*mults[c]
                                out_graph[i].add(k)
                                in_graph[k].add(i)
                                out_graph_type[i][k] = set([type_r])
                            else:
                                edges[c][k] = prod_dict[j][k]*mults[c]
            for c in range(l_c):
                graphs[c].append(edges[c])
        else:
            for c in range(l_c):
                graphs[c].append({})
    avg_graph = []
    for i in range(spec_len):
        avg = {}
        for c in range(l_c):
            for j in graphs[c][i]:
                if j in avg:
                    avg[j]+=graphs[c][i][j]
                else:
                    avg[j] = graphs[c][i][j]
        for x in avg:
            avg[x]*=1/l_c
        avg_graph.append(avg)
        
    graph_mat_lil = lil_matrix((spec_len,spec_len), dtype=None)

    edges = []
    for i in range(len(out_graph)):
        for j in out_graph[i]:
            edges.append((i,j))

    count = 0
    for i,j in edges:
        graph_mat_lil[i,j] = 1

    graph_mat = graph_mat_lil.tocsr(copy=False)

    scc_result = connected_components(graph_mat, directed=True, connection='strong', return_labels=True)

    scc_dic = {}
    for i in range(len(scc_result[1])):
        if scc_result[1][i] in scc_dic:
            scc_dic[scc_result[1][i]].append(i)
        else:
            scc_dic[scc_result[1][i]] = [i]
            
    scc = []
    for i in scc_dic:
        if len(scc_dic[i])>1:
            scc.append(scc_dic[i])

    distance,preds,root_specs = dijkstra(graph_mat, directed=True, indices=roots, return_predecessors=True, unweighted=True, limit=np.inf, min_only=True)    
    #return distance, preds, root_specs
    # measure scc's
    
    scc_lens = [len(i) for i in scc]
    print('6.5')
    # seeing how many species can be joined by the common parent rule
    
    #identifying species with same parent, same incoming reaction types and same outgoing reaction types
    
    print('7')
    #4 
    print('9')
    paths = []
    path_roots = []
    for i in range(spec_len):
        if distance[i]<100:
            if preds[i]<0:
                path = []
            else:
                marker = preds[i]
                path = [marker]
                counter = 0
                while marker not in roots and counter<100:
                    marker = preds[marker]
                    path.append(marker)
            paths.append(path)
            path_roots.append(root_specs[i])
        else:
            paths.append([])
            path_roots.append('none')

    paths_with_types = []
    for i in range(len(paths)):
        path = []
        count = 0
        for j in paths[i]:
            if count == 0:
                path.append(in_graph_type[i][j])
            else:
                path.append(in_graph_type[paths[i][count-1]][j])
            count = count + 1
        path.insert(0,path_roots[i])
        paths_with_types.append(path)
    print('10')
    path_type_strings = []
    for i in paths_with_types:
        string = str(i[0]) + '+'
        for j in i[1:]:
            for k in j:
                string = string + str(k) + ','
            string = string+'+'
        path_type_strings.append(string)

    path_sim_dict = {}
    for i in range(len(path_type_strings)):
        if path_type_strings[i] in path_sim_dict:
            path_sim_dict[path_type_strings[i]].append(i)
        else:
            path_sim_dict[path_type_strings[i]]= [i]
    
    groups = []  
    for i in path_sim_dict:
        if len(path_sim_dict[i])>1:
            groups.append(path_sim_dict[i])
    print('11')
    in_cycle_specs = []
    scc_out_specs = []
    for i in scc:
        count = count + 1
        #if count%100 == 0:
         #   print(count)
        in_cycle_spec = set()
        out_spec = set()
        for p in i:
            for k in in_graph[p]:
                if k not in i:
                    in_cycle_spec.add(p)
            for k in out_graph[p]:
                if k not in i:
                    out_spec.add(k)
        in_cycle_specs.append(in_cycle_spec)
        scc_out_specs.append(out_spec)
    
    #return graph, in_graph, out_graph, scc_out_specs, in_cycle_specs, scc
    scc_set = set()
    scc_dict_2 = {}
    for i in range(len(scc)):
        for j in scc[i]:
            scc_dict_2[j] = i
            scc_set.add(j)
        
    county = 0
    yields = []
    cycle_out_graphs = [{} for c in range(l_c)]
    for c in range(l_c):
        print(county)
        county +=1
        
       # new_graph, new_in, new_out = copy_graph(graphs[c], in_graph, out_graph)
        
        new_graph = []
        for i in graphs[c]:
            dicy = {}
            for j in i:
                dicy[j] = i[j]
            new_graph.append(dicy)


        new_in = []
        for i in in_graph:
            sety = set()
            for j in i:
                sety.add(j)
            new_in.append(sety)

        new_out = []
        for i in out_graph:
            sety = set()
            for j in i:
                sety.add(j)
            new_out.append(sety)

        for i in range(len(scc)):
            counte = 0
            for x in in_cycle_specs[i]:
                counte +=1
                new_out[x] = scc_out_specs[i].union(set(scc[i]))
                leny = int(np.sqrt(len(scc[i])))
                data, in_cyc_data = cycle_simulator_3(scc[i], x, graphs[c],out_graph,in_graph, [leny+40,2*(leny+40)],1e-6,set(scc[i]).union(scc_out_specs[i]),scc_out_specs[i])
                cycle_out_graphs[c][x] = data
                ful_dat = {}
                for p in data:
                    ful_dat[p] = data[p]
                #for p in in_cyc_data:
                 #   ful_dat[p] = in_cyc_data[p]
                new_graph[x] = ful_dat
            for y in scc[i]:
                if y not in in_cycle_specs[i]:

                    for s in out_graph[y]:
                        if y in new_in[s]:
                            new_in[s].remove(y)
                    new_in[y] = set(in_cycle_specs[i])
                    new_out[y] = set()
                    new_graph[y] = set()

        rooted = []
        
        for i in roots:
            rooted.append([i,0.001])
        yi = get_yields(roots,1e-8, new_graph, new_in, new_out, back_set,scc,scc_dict_2, scc_set, spec_len)
        yields.append(yi)
    return yields, new_graph, scc


def get_mech_yields(mechanism, background_spc_n, conditions, settings):
    #creating necessary variables/objects
    
    needed_concentrations = set()
    test_reactions = []
    not_rxns = set(settings['Background Rxns'])
    for i in settings['Aerosol Rxns']:
        not_rxns.add(i)
    for i in range(len(mechanism.reactions)):
        if i not in not_rxns:
            '''r = mechanism.reactions[i].reactants
            p = mechanism.reactions[i].prod_dict
            k1 = mechanism.reactions[i].rate_law
            k2 = mechanism.reactions[i].eval_rate_law
            k3 = mechanism.reactions[i].rate
            k4 = mechanism.reactions[i].rate_string
            test_reactions.append(Reaction(r,p,k1,k2,k3,k4))'''
            test_reactions.append(deepcopy(mechanism.reactions[i]))
     
    test_mechanism = Mechanism(deepcopy(mechanism.species),test_reactions)
  
    reac_len = len(test_mechanism.reactions)

    spec_len = len(test_mechanism.species)
    dic = {test_mechanism.species[i]:i for i in range(spec_len)}

    
    #no_group = [dic[i] for i in settings['No Group']]

    species_list = [i for i in range(spec_len)]
    roots = [species_list_names.index(i) for i in settings['roots']]
    l_c = len(conditions)
    back_set = set()
    for i in background_spc_n:
        if i in species_list_names:
            back_set.add(species_list_names.index(i))
        else:
            back_set.add(i)
    prod_coeff_list = []
    prod_dict = [{} for i in range(reac_len)]
    reac_list_n = []
    prod_list_n = []
    reac_len_new = deepcopy(reac_len)
    for i in range(reac_len):
        test_mechanism.reactions[i].rate = []
        reac_list_n.append(test_mechanism.reactions[i].reactants)
        prod_list_n.append(list(test_mechanism.reactions[i].prod_dict))
        for k in test_mechanism.reactions[i].prod_dict:
            prod_dict[i][dic[k]] = test_mechanism.reactions[i].prod_dict[k]


    reac_list, prod_list, reac_no_back, prod_no_back, rxn_reac, rxn_prod = rxn_index_convert(reac_list_n,prod_list_n,background_spc_n, background_spc, reac_len, spec_len, dic)

    
    rxn_prod = [set(i) for i in rxn_prod]
    rxn_reac = [set(i) for i in rxn_reac]

    #measuring rates for all conditions
    for i in conditions: 
        M = pressure_to_m(i['pressure'],i['temp'])
        TEMP = i['temp']
        SUN = i['sun']
        CFACTOR = 2.5e+19
        p_fac = M/1000000000
        double_rxns = []
        count = 0
        for j in test_mechanism.reactions:
            if j.rate_law!='null':
                if 'J' in j.rate_law:
                    j.eval_rate_law = solve_j_rate(j.rate_law, i['sza'],i['sun'])
                else:
                    j.eval_rate_law = eval(j.rate_law)
                if len(j.reactants)==1:
                    j.rate.append(j.eval_rate_law)

                elif j.reactants[1] in background_spc_n and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[1]]*p_fac)

                elif j.reactants[0] in background_spc_n and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[0]]*p_fac)

                elif j.reactants[1] in i and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[1]]*p_fac)

                elif j.reactants[0] in i and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law*i[j.reactants[0]]*p_fac)

                else:
                    needed_concentrations.add(j.reactants[1])
                    root_conc = 0
                    for r in settings['roots']:
                        root_conc = i[r] + root_conc
                    j.rate.append(j.eval_rate_law*0.005*root_conc*p_fac)
                    double_rxns.append(count)
                count = count + 1
            else:
                counted_reactants = 0
                for s in j.reactants:
                    if s not in settings['No Counts']:
                        counted_reactants+=1
                if counted_reactants==1:
                    j.rate.append(j.eval_rate_law[c_count])
                elif j.reactants[1] in background_spc_n and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in background_spc_n and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[0]]*p_fac)
                elif j.reactants[1] in i and j.reactants[1] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[1]]*p_fac)
                elif j.reactants[0] in i and j.reactants[0] not in settings['roots']:
                    j.rate.append(j.eval_rate_law[c_count]*i[j.reactants[0]]*p_fac)
                else:
                    needed_concentrations.add(j.reactants[1])
                    root_conc = 0
                    for r in settings['roots']:
                        root_conc = i[r] + root_conc
                    j.rate.append(j.eval_rate_law[c_count]*0.05*root_conc*p_fac)

                    double_rxns.append(count)

    # identifying weak reactions which can be removed
    '''if settings['Remove weak reactions'] == True:
        keep_rxns = set()
        for i in range(spec_len):
            rate_list = []
            rate_dict = {}
            for j in rxn_reac[i]:
                dat = max(test_mechanism.reactions[j].rate)
                rate_list.append(dat)
                rate_dict[j] = dat
            if len(rate_list)>0:
                max_r = max(rate_list)
            else:
                max_r = 0
            for k in rate_dict:
                if rate_dict[k]>max_r*settings['Weak rxn cutoff']:
                    keep_rxns.add(k)
                #else:
                    #print(k, rate_dict[k], max_r, test_mechanism.reactions[k].rate)
        remove_rxns = set(range(reac_len)) - keep_rxns'''
        
    # how many species can we get rid of from the weak rxns?


        

    graphs = [[] for c in range(l_c)]
    in_graph = [set() for i in range(spec_len)]
    out_graph = [set() for i in range(spec_len)]
    out_graph_type = [{} for i in range(spec_len)]
    in_graph_type  = [{} for i in range(spec_len)]

    
    rxn_types = {}
    
    for i in range(spec_len):
        if i not in back_set:
            edges = [{} for c in range(l_c)]
            rate_sums = [0 for c in range(l_c)]
            

            for j in rxn_reac[i]:
                for c in range(l_c):
                    rate_sums[c] = rate_sums[c] + test_mechanism.reactions[j].multiplier*test_mechanism.reactions[j].rate[c]

            for j in rxn_reac[i]:
                mults = [0 for c in range(l_c)]
                for c in range(l_c):
                    #print(test_mechanism.reactions[j].multiplier)
                    mults[c] = test_mechanism.reactions[j].multiplier*test_mechanism.reactions[j].rate[c]/max(1e-20,rate_sums[c])
                if len(test_mechanism.reactions[j].reactants) == 1:
                    type_r = 'solo'
                else:
                    mark = True
                    for p in reac_list[j]:
                        if p in background_spc:
                            type_r = p
                            mark = False
                    if mark:
                        type_r = 'double'
                rxn_types[j] = type_r

                for k in prod_dict[j]:
                    if i in in_graph_type[k]:
                        in_graph_type[k][i].add(type_r)
                    else:
                        in_graph_type[k][i] = set([type_r])
                    for c in range(l_c):
                        if k in edges[c]:
                            if c==0:
                                edges[c][k] = edges[c][k] + prod_dict[j][k]*mults[c]
                                out_graph[i].add(k)
                                in_graph[k].add(i)
                                out_graph_type[i][k].add(type_r)
                            else:
                                edges[c][k] = edges[c][k] + prod_dict[j][k]*mults[c]

                        
                        else:
                            if c==0:
                                edges[c][k] = prod_dict[j][k]*mults[c]
                                out_graph[i].add(k)
                                in_graph[k].add(i)
                                out_graph_type[i][k] = set([type_r])
                            else:
                                edges[c][k] = prod_dict[j][k]*mults[c]
                
            for c in range(l_c):
                graphs[c].append(edges[c])
        else:
            for c in range(l_c):
                graphs[c].append({})
    avg_graph = []
    for i in range(spec_len):
        avg = {}
        for c in range(l_c):
            for j in graphs[c][i]:
                if j in avg:
                    avg[j]+=graphs[c][i][j]
                else:
                    avg[j] = graphs[c][i][j]
        for x in avg:
            avg[x]*=1/l_c
        avg_graph.append(avg)
    
    graph_mat_lil = lil_matrix((spec_len,spec_len), dtype=None)

    edges = []
    for i in range(len(out_graph)):
        for j in out_graph[i]:
            edges.append((i,j))
    count = 0
    for i,j in edges:
        graph_mat_lil[i,j] = 1

    graph_mat = graph_mat_lil.tocsr(copy=False)

    scc_result = connected_components(graph_mat, directed=True, connection='strong', return_labels=True)

    scc_dic = {}
    for i in range(len(scc_result[1])):
        if scc_result[1][i] in scc_dic:
            scc_dic[scc_result[1][i]].append(i)
        else:
            scc_dic[scc_result[1][i]] = [i]
            
    scc = []
    for i in scc_dic:
        if len(scc_dic[i])>1:
            scc.append(scc_dic[i])

    distance,preds,root_specs = dijkstra(graph_mat, directed=True, indices=roots, return_predecessors=True, unweighted=True, limit=np.inf, min_only=True)    
    #return distance, preds, root_specs
    # measure scc's

    scc_lens = [len(i) for i in scc]

    # seeing how many species can be joined by the common parent rule
    
    #identifying species with same parent, same incoming reaction types and same outgoing reaction types
    
    
    paths = []
    path_roots = []
    for i in range(spec_len):
        if distance[i]<100:
            if preds[i]<0:
                path = []
            else:
                marker = preds[i]
                path = [marker]
                counter = 0
                while marker not in roots and counter<100:
                    marker = preds[marker]
                    path.append(marker)
            paths.append(path)
            path_roots.append(root_specs[i])
        else:
            paths.append([])
            path_roots.append('none')

    paths_with_types = []
    for i in range(len(paths)):
        path = []
        count = 0
        for j in paths[i]:
            if count == 0:
                path.append(in_graph_type[i][j])
            else:
                path.append(in_graph_type[paths[i][count-1]][j])
            count = count + 1
        path.insert(0,path_roots[i])
        paths_with_types.append(path)


    path_type_strings = []
    for i in paths_with_types:
        string = str(i[0]) + '+'
        for j in i[1:]:
            for k in j:
                string = string + str(k) + ','
            string = string+'+'
        path_type_strings.append(string)

    path_sim_dict = {}
    for i in range(len(path_type_strings)):
        if i not in back_set:
            if path_type_strings[i] in path_sim_dict:
                path_sim_dict[path_type_strings[i]].append(i)
            else:
                path_sim_dict[path_type_strings[i]]= [i]
    groups = []  
    for i in path_sim_dict:
        if len(path_sim_dict[i])>1:
            groups.append(path_sim_dict[i])
    #print('out 31', out_graph[31])  
    #for i in groups:

    in_cycle_specs = []
    scc_out_specs = []
    for i in scc:
        count = count + 1
        #if count%100 == 0:
         #   print(count)
        in_cycle_spec = set()
        out_spec = set()
        for p in i:
            for k in in_graph[p]:
                if k not in i:
                    in_cycle_spec.add(p)
            for k in out_graph[p]:
                if k not in i:
                    out_spec.add(k)
        in_cycle_specs.append(in_cycle_spec)
        scc_out_specs.append(out_spec)
        
    
    
    #return graph, in_graph, out_graph, scc_out_specs, in_cycle_specs, scc
    scc_set = set()
    scc_dict_2 = {}
    for i in range(len(scc)):
        for j in scc[i]:
            scc_dict_2[j] = i
            scc_set.add(j)
            

    
    #print('groups', groups)

    county = 0
    yields = []
    cycle_out_graphs = [{} for c in range(l_c)]
    for c in range(l_c):
        #print(county)
        county +=1
        
       # new_graph, new_in, new_out = copy_graph(graphs[c], in_graph, out_graph)
        
        new_graph = []
        for i in graphs[c]:
            dicy = {}
            for j in i:
                dicy[j] = i[j]
            new_graph.append(dicy)


        new_in = []
        for i in in_graph:
            sety = set()
            for j in i:
                sety.add(j)
            new_in.append(sety)

        new_out = []
        for i in out_graph:
            sety = set()
            for j in i:
                sety.add(j)
            new_out.append(sety)

        
        for i in range(len(scc)):
            counte = 0
            
            for x in in_cycle_specs[i]:
                counte +=1
                new_out[x] = scc_out_specs[i].union(set(scc[i]))
                leny = int(np.sqrt(len(scc[i])))
                data, in_cyc_data = cycle_simulator_3(scc[i], x, graphs[c],out_graph,in_graph, [leny+40,2*(leny+40)],1e-6,set(scc[i]).union(scc_out_specs[i]),scc_out_specs[i])
                cycle_out_graphs[c][x] = data
                ful_dat = {}
                for p in data:
                    ful_dat[p] = data[p]
                for p in in_cyc_data:
                    ful_dat[p] = in_cyc_data[p]
                new_graph[x] = ful_dat

            for y in scc[i]:
                if y not in in_cycle_specs[i]:

                    for s in out_graph[y]:
                        if y in new_in[s]:
                            new_in[s].remove(y)
                    new_in[y] = set(in_cycle_specs[i])
                    new_out[y] = set()
                    new_graph[y] = set()

        
        yi = get_yields(roots, 0, new_graph, new_in, new_out, back_set,scc,scc_dict_2, scc_set, spec_len)

        yields.append(yi)
    return yields #, graphs, scc  


#create_f0am_file(network,reaction_list,species_list,name)
def create_f0am_file(mech,name):
    spec_2_add = "SpeciesToAdd = {'ISOPN'; "
    count = 0
    for i in mech.species:
        count+=1
        if count%1000 == 0:
            print(count)
        spec_2_add = spec_2_add + "'" + i +"'"+ ';'
    spec_2_add = spec_2_add[:-1]   
    spec_2_add = spec_2_add  + "};"
   
    eq_str = ''
    for i in range(len(mech.reactions)):
        r_string = ''
        for j in mech.reactions[i].reactants:
            r_string = r_string + str(j) + ' + '
        r_string = r_string[:-2] + '= '
        for j in mech.reactions[i].prod_dict:
            r_string = r_string + str(j) + ' + '
        r_string = r_string[:-3]
        reac_str = ''
        reac_str = reac_str + "\ni=i+1;\nRnames{i} = '" + r_string + "';\nk(:,i) = "+ str(mech.reactions[i].rate_law)+ '*'+str(mech.reactions[i].multiplier) + ';\n'
        counter = 1
        for j in mech.reactions[i].reactants:
            reac_str = reac_str + 'Gstr{i,'+str(counter)+"} = '"+str(j)+"'; "
            counter = counter + 1
        reac_str = reac_str +'\n'
        for k in mech.reactions[i].reactants:
            reac_str = reac_str + 'f'+ str(k) +'(i)'+'='+'f'+ str(k) +'(i)'+'-1' + '; '
        for k in mech.reactions[i].prod_dict:
            reac_str = reac_str + 'f'+ str(k)+'(i)'+'='+'f'+ str(k)+'(i)'+'+'+str(mech.reactions[i].prod_dict[k]) + '; '
        
        reac_str = reac_str +'\n'
        reac_str = reac_str +'\n'
        eq_str = eq_str + reac_str

        
    #eq_str = eq_str.replace('CH2O','HCHO')
    eq_str = eq_str.replace('.*RO2',';%.*RO2')
    eq_str = eq_str.replace('J(22)','J22')
    eq_str = eq_str.replace('J(34)','J34')
    eq_str = eq_str.replace('J(41)','J41')
    eq_str = eq_str.replace('ALK(','F0AM_isop_ALK(T,M,')
    eq_str = eq_str.replace('EPO(','F0AM_isop_EPO(T,M,')
    eq_str = eq_str.replace('NIT(','F0AM_isop_NIT(T,M,')
    eq_str = eq_str.replace('TROE(','F0AM_isop_TROE2(T,M,')
    eq_str = eq_str.replace('TUN(','F0AM_isop_TUN(T,M,')
    eq_str = eq_str.replace('ISO1(','F0AM_isop_ISO1(T,')
    eq_str = eq_str.replace('ISO2(','F0AM_isop_ISO2(T,')
    eq_str = eq_str.replace('KCO(','F0AM_isop_KCO(T,M,')
    eq_str = eq_str.replace('FALL(','F0AM_isop_FALL(T,M,')
    eq_str = eq_str.replace('TEMP','T')
    eq_str = eq_str.replace('**','.^')
    eq_str = eq_str.replace('EXP','exp')
    eq_str = eq_str.replace('*','.*')
    eq_str = eq_str.replace('.*O2','.*0.21')
    eq_str = eq_str.replace('KAPHO2(T)','KAPHO2')
    eq_str = eq_str.replace('KFPAN(T,M)','KFPAN')
    eq_str = eq_str.replace('KNO3AL(T)','KNO3AL')
    eq_str = eq_str.replace('KAPNO(T)','KAPNO')
    eq_str = eq_str.replace('KBPAN(T,M)','KBPAN')
    eq_str = eq_str.replace('KRO2NO3()','KRO2NO3')
    full = spec_2_add + '\n'+'RO2ToAdd = {};'+'\n'+'AddSpecies'+'\n'+ eq_str
    f0am_file = open("f0am_"+name+".m","w+")
    f0am_file.write(full)


def red_mechanism_to_graph(mechy, conditions, spec_len, red_specs, back_set):
    l_c = len(conditions)
    red_graphs = [[{} for i in range(spec_len)] for c in range(l_c)]
    red_in_graph = [set() for i in range(spec_len)]
    red_out_graph = [set() for i in range(spec_len)]

    red_rxn_reac = {}
    red_spec_dict = {}
    for i in range(len(red_specs)):
        red_spec_dict[red_specs[i]] = i
    for i in range(len(mechy.reactions)):
        for j in mechy.reactions[i].reactants:
            if j not in back_set:
                if j in red_rxn_reac:  
                    red_rxn_reac[j].add(i)
                else:
                    red_rxn_reac[j] = {i}

    
    for i in red_rxn_reac:
        if i not in back_set:
            edges = [{} for c in range(l_c)]
            rate_sums = [0 for c in range(l_c)]
            
            for j in red_rxn_reac[i]:

                for c in range(l_c):
                    rate_sums[c] = rate_sums[c] + mechy.reactions[j].rate[c]*mechy.reactions[j].multiplier
            for j in red_rxn_reac[i]:
                mults = [0 for c in range(l_c)]
                for c in range(l_c):
                    mults[c] = mechy.reactions[j].rate[c]*mechy.reactions[j].multiplier/max(1e-20,rate_sums[c])
                if len(mechy.reactions[j].reactants) == 1:
                    type_r = 'solo'
                else:
                    mark = True
                    for p in mechy.reactions[j].reactants:
                        if p in background_spc:
                            type_r = p
                            mark = False
                    if mark:
                        type_r = 'double'
                for k in mechy.reactions[j].prod_dict:

                    #if i in in_graph_type[k]:
                     #  in_graph_type[k][i].add(type_r)
                    #else:
                    #    in_graph_type[k][i] = set([type_r])
                    for c in range(l_c):
                        if k in edges[c]:
                            if c==0:
                                edges[c][k] = edges[c][k] + mechy.reactions[j].prod_dict[k]*mults[c]
                                red_out_graph[i].add(k)
                                red_in_graph[k].add(i)
                                #red_out_graph_type[i][k].add(type_r)
                            else:
                                edges[c][k] = edges[c][k] + mechy.reactions[j].prod_dict[k]*mults[c]

                        
                        else:
                            if c==0:
                                edges[c][k] = mechy.reactions[j].prod_dict[k]*mults[c]
                                red_out_graph[i].add(k)
                                red_in_graph[k].add(i)
                                #red_out_graph_type[i][k] = set([type_r])
                            else:
                                edges[c][k] = mechy.reactions[j].prod_dict[k]*mults[c]
            for c in range(l_c):
                red_graphs[c][i] = edges[c]
    return red_graphs, red_in_graph, red_out_graph, red_spec_dict



def red_mechanism_to_graph_2(mechy, conditions, spec_len, red_specs, back_set, scc):
    l_c = len(conditions)
    red_graphs = [[{} for i in range(spec_len)] for c in range(l_c)]
    red_in_graph = [set() for i in range(spec_len)]
    red_out_graph = [set() for i in range(spec_len)]
    T1 = 0
    T2 = 0
    T3 = 0
    red_rxn_reac = {}
    red_spec_dict = {}
    scc_dup = deepcopy(set(scc))
    for i in range(len(mechy.reactions)):
        rxn_specs = set(mechy.reactions[i].reactants)
        if any([f in scc for f in rxn_specs]):
            for j in rxn_specs:
                if j not in back_set:
                    scc_dup.add(j)
    for i in range(len(red_specs)):
        red_spec_dict[red_specs[i]] = i
    
    for i in range(len(mechy.reactions)):
        rxn_specs = set(mechy.reactions[i].reactants)
        for l in mechy.reactions[i].prod_dict:
            rxn_specs.add(l)
        if any([f in scc_dup for f in rxn_specs]):
            for j in mechy.reactions[i].reactants:
                if j not in back_set:
                    if j in red_rxn_reac:  
                        red_rxn_reac[j].add(i)
                    else:
                        red_rxn_reac[j] = {i}

    
    for i in red_rxn_reac:
        t1 = time.time()
        edges = [{} for c in range(l_c)]
        rate_sums = [0 for c in range(l_c)]
        
        for j in red_rxn_reac[i]:

            for c in range(l_c):
                rate_sums[c] = rate_sums[c] + mechy.reactions[j].rate[c]*mechy.reactions[j].multiplier
        
        for j in red_rxn_reac[i]:
            t15 = time.time()
            mults = [0 for c in range(l_c)]
            for c in range(l_c):
                mults[c] = mechy.reactions[j].rate[c]*mechy.reactions[j].multiplier/max(1e-20,rate_sums[c])
            t2 = time.time()
            #if i==10:
            #    print('MORE 10 Stuff', mechy.reactions[j].reactants, mechy.reactions[j].prod_dict)
            for k in mechy.reactions[j].prod_dict:

                #if i in in_graph_type[k]:
                 #  in_graph_type[k][i].add(type_r)
                #else:
                #    in_graph_type[k][i] = set([type_r])
                for c in range(l_c):
                    if k in edges[c]:
                        if c==0:
                            edges[c][k] = edges[c][k] + mechy.reactions[j].prod_dict[k]*mults[c]
                            red_out_graph[i].add(k)
                            red_in_graph[k].add(i)
                            #red_out_graph_type[i][k].add(type_r)
                        else:
                            edges[c][k] += mechy.reactions[j].prod_dict[k]*mults[c]

                    
                    else:
                        if c==0:
                            edges[c][k] = mechy.reactions[j].prod_dict[k]*mults[c]
                            red_out_graph[i].add(k)
                            red_in_graph[k].add(i)
                            #red_out_graph_type[i][k] = set([type_r])
                        else:
                            edges[c][k] = mechy.reactions[j].prod_dict[k]*mults[c]
            t3 = time.time()
            T2 += t2-t15
            T3 += t3-t2
        for c in range(l_c):
            red_graphs[c][i] = edges[c]
        
        T1 += t15-t1
       
    #print('times',T1, T2, T3)
    return red_graphs, red_in_graph, red_out_graph, red_spec_dict



def update_red_mechanism_graph(mechy, conditions, spec_len, red_specs, back_set, rxn, red_graphs_2, red_in_graph_2, red_out_graph_2, red_spec_dict_2, short_graph):
    l_c = len(conditions)
    red_graphs = [[{} for i in range(spec_len)] for c in range(l_c)]
    for c in range(l_c):
        for j in short_graph[c]:
            red_graphs[c][j] = deepcopy(short_graph[c][j])
    red_in_graph = red_in_graph_2
    red_out_graph = red_out_graph_2
    red_spec_dict = red_spec_dict_2
    
    rxn_specs = []
    for i in mechy.reactions[rxn].reactants:
        if i not in back_set:
            rxn_specs.append(i)
            #print('RXN', rxn, i)


    for re in rxn_specs:
        red_rxn_reac = set()
        for i in range(len(mechy.reactions)):
            for j in mechy.reactions[i].reactants:
                if j == re:
                    red_rxn_reac.add(i)
                    
    
        edges = [{} for c in range(l_c)]
        rate_sums = [0 for c in range(l_c)]
        for j in red_rxn_reac:
            for c in range(l_c):
                rate_sums[c] = rate_sums[c] + mechy.reactions[j].rate[c]*mechy.reactions[j].multiplier
        for j in red_rxn_reac:
    
            
            
            mults = [0 for c in range(l_c)]
            for c in range(l_c):
                mults[c] = mechy.reactions[j].rate[c]*mechy.reactions[j].multiplier/max(1e-20,rate_sums[c])
                if re == 10:
                    pass
                    #print('MILT', mults[0], mechy.reactions[j].prod_dict)
            for k in mechy.reactions[j].prod_dict:
                #if k==31:
                    #print('HCHO',mechy.reactions[j].prod_dict[k], mults[c])
                for c in range(l_c):
                   
                    if k in edges[c]:
                        if c==0:
                            edges[c][k] = edges[c][k] + mechy.reactions[j].prod_dict[k]*mults[c]
                        else:
                            edges[c][k] += mechy.reactions[j].prod_dict[k]*mults[c]

                    
                    else:
                        if c==0:
                            edges[c][k] = mechy.reactions[j].prod_dict[k]*mults[c]
                        else:
                            edges[c][k] = mechy.reactions[j].prod_dict[k]*mults[c]
        #print('EDGES', re, edges[0])
        for c in range(l_c):
            for f in red_graphs[c][re]:
                if f in edges[c]:
                    red_graphs[c][re][f]=edges[c][f]
                else:
                    print('no edge', f)
                #red_graphs[c][re] = edges[c]
        

    return red_graphs, red_in_graph, red_out_graph, red_spec_dict

def cycle_simulator_3_modified(cycle_species, in_specs, graph, out_graph, in_graph, iteration_set, cutoff, all_spec, out_spec):
    data_set = []
    search = {}
    start_tot = 0
    for i in in_specs:
        search[i[0]] = i[1]
        start_tot += i[1]
    counter = 0
    data = {j:0 for j in all_spec}
    out_dat = {j:0 for j in cycle_species}
    while counter<iteration_set[1]:
        new_search = {}
        if counter == iteration_set[0]:
            data_1 = {p:data[p] for p in data}
            search_1 = {p:search[p] for p in search}
        for k in search:
            #data[k] = data[k]-search[k]
            return_amount = 0
            for x in graph[k]:
                data[x] = data[x] + graph[k][x]*search[k]
                
                if x in cycle_species:
                    return_amount = return_amount + search[k]*graph[k][x]
                    if graph[k][x]*search[k]>cutoff:
                        if x in new_search:
                            new_search[x] = new_search[x] + graph[k][x]*search[k]
                        else:
                            new_search[x] = graph[k][x]*search[k]
            out_dat[k] = out_dat[k] + search[k]-return_amount
        search = {p:new_search[p] for p in new_search}
        counter = counter + 1
    frac1 = 0
    frac2 = 0
    for n in cycle_species:
        if n in search_1:
            frac1 = frac1 + search_1[n]
        if n in search:
            frac2 = frac2 + search[n]
    frac1 = frac1/max(1e-20,start_tot)
    frac2 = frac2/max(1e-20,start_tot)
    for i in out_dat:
        out_dat[i] = max(0,out_dat[i]/max(1-frac2,1e-20))
    final_data = {}
    within_data = {}
    denom = frac2-frac1
    if denom ==0:
        denom = 1e-20
    for n in out_spec:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        final_data[n] = dat
    for n in cycle_species:
        slope = (data[n]-data_1[n])/denom
        dat = data[n] - slope*frac2
        within_data[n] = dat
    within_data_2 = {}
    within_data_sum = 0
    for i in within_data:
        within_data_sum = within_data_sum + within_data[i]
    for i in within_data:
        within_data_2[i] = abs(within_data[i]/max(within_data_sum,1e-20))

    final_in_data = {}
    for i in within_data:
        final_in_data[i] = np.mean([within_data[i],out_dat[i]])
    return final_data, within_data, out_dat#, data


def weighted_median(df, val, weight):
    df_sorted = df.sort_values(val)
    cumsum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum() / 2.
    return df_sorted[cumsum >= cutoff][val].iloc[0]


# grad descent func: ([param list with starting values], evaluation function)
# evaluate the function for each parameter for each step
def grad_descent_cycles(m_para, m_para_map, mechy, ref_data, data_map, cycle_info, settings, conditions, red_specs, back_set, spec_len, scc):
    #cycle_info = cycle species, in species, all species, out species
    #print('MPARA MAP',m_para_map)
    print('len mpara', len(m_para))
    count = 0
    '''new_m_para = {}
    for i in m_para:
        if count==itybity:
            new_m_para[i] = m_para[i]
        count+=1
    m_para = deepcopy(new_m_para)'''
    working_mech = deepcopy(mechy)
    score_log = []
    learn = settings['learn_rate']
    for p in m_para:
        if m_para[p]==0:
            m_para[p]==0.001
    #T1 = 0
    #T2 = 0
    #T3 = 0
    #T4 = 0
    #T5 = 0
    #print(' data mpa', data_map
    print(m_para)
    m_para_sum = sum(m_para.values())
    for steps in range(settings['steps']):
        #print(m_para)
        #%memit
        #print('step 1')
        #measure the current score having done all of the changes from measuring the gradient
        current_scores = []
        for p in m_para:
            if len(m_para_map[p])==1:
                working_mech.reactions[m_para_map[p][0]].multiplier = max(0,m_para[p])
            elif len(m_para_map[p])==2:
                working_mech.reactions[m_para_map[p][0]].prod_dict[m_para_map[p][1]] = m_para[p]
        #print('step 2')
        red_graphs, red_in_graph, red_out_graph, red_spec_dict = red_mechanism_to_graph_2(working_mech, conditions, spec_len, red_specs, back_set, scc)
        short_red_graphs = []
        for x in red_graphs:
            new_g = {}
            for j in range(len(x)):
                if x[j]!={}:
                    new_g[j] = x[j]
            short_red_graphs.append(new_g)
        for c in range(len(conditions)):
            data = cycle_simulator_3_modified(cycle_info[0], cycle_info[1][c], red_graphs[c], red_out_graph, red_in_graph, [20,40], 1e-6, cycle_info[2], cycle_info[3]) ###
            score = get_cycle_score(data, ref_data[c], data_map)
            if steps%40==0 and c==0:
                print(data, ref_data[c])
            current_scores.append(score)
        current_score = np.mean(current_scores)
        #print('step 3')
        print(steps, current_score)
        score_log.append(current_score)
        if score_log[steps] == min(score_log):
            best_mech = deepcopy(working_mech)
        # take a step for each parameter to measure gradient, then change each parameter accordingly
        new_m_para = deepcopy(m_para)
        #print('step 4')
        for p in m_para:
            
            #%memit
            #print('step 5')
            #x1 = time.time()
            working_mech_2 = []
            for rex in working_mech.reactions:
                rxn = Reaction(deepcopy(rex.reactants), deepcopy(rex.prod_dict), rex.rate_law, rex.eval_rate_law, rex.rate, rex.rate_string, deepcopy(rex.multiplier))
                working_mech_2.append(deepcopy(rxn))
            #working_mech_2 = deepcopy(working_mech.reactions)
            #x2 = time.time()
            #T1 += x2-x1
            scores = []
            test_m_para = deepcopy(m_para)
            #print('original value', working_mech_2[m_para_map[p][0]].multiplier)
            if test_m_para[p] !=0:
                test_m_para[p] = settings['grad_step']*test_m_para[p] #settings['grad_step']
            else:
                test_m_para[p] = 0.001
            #test_m_para[p] = test_m_para[p] + 0.0000001

            if len(m_para_map[p])==1:
                working_mech_2[m_para_map[p][0]].multiplier = deepcopy(test_m_para[p])
            if len(m_para_map[p])==2:
                working_mech_2[m_para_map[p][0]].prod_dict[m_para_map[p][1]] = deepcopy(test_m_para[p])

            #print('updfated value', working_mech_2[m_para_map[p][0]].multiplier)
            work_mech_2_full = Mechanism(working_mech.species, working_mech_2)
            #x3 = time.time()
            #test_graphs_x, red_in_graph_x, red_out_graph_x, red_spec_dict_x = red_mechanism_to_graph_2(work_mech_2_full, conditions, spec_len, red_specs, back_set, scc)
            test_graphs, red_in_graph, red_out_graph, red_spec_dict = update_red_mechanism_graph(work_mech_2_full, conditions, spec_len, red_specs, back_set, m_para_map[p][0], red_graphs, red_in_graph, red_out_graph, red_spec_dict, short_red_graphs)
            #if 10 in work_mech_2_full.reactions[m_para_map[p][0]].reactants:
            #    print('10 stuff', test_graphs[0][10], test_graphs_x[0][10])

            '''for z in range(len(test_graphs_x[0])):
                for y in test_graphs_x[0][z]:
                    #pass
                    #print(test_graphs[0][z][y], test_graphs_x[0][z][y], red_graphs[0][z][y])
                    if y in test_graphs[0][z] and test_graphs_x[0][z][y] != test_graphs[0][z][y]:
                        #$print('NOT SAME', z,y, test_graphs[0][z][y], test_graphs_x[0][z][y], red_graphs[0][z][y])
                        #else:
                        #    print('THE SAME', z, y)
                        pass'''
            #x4 = time.time()
            for c in range(len(conditions)):
                data = cycle_simulator_3_modified(cycle_info[0], cycle_info[1][c], test_graphs[c], red_out_graph, red_in_graph, [20,40], 1e-6, cycle_info[2], cycle_info[3]) ###
                score = get_cycle_score(data, ref_data[c], data_map) ###
                #T2 += x4-x3
                scores.append(score)
            #x5 = time.time()
            #print('step 6')
            param_score = np.mean(scores)
            
            #print(np.mean(scores), p, working_mech_2[m_para_map[p][0]].multiplier)
            #print(param_score, current_score)
            # this is the gradient descent new step function
            # make sure this is right...

            #print('h', p, m_para[p],(param_score - current_score)/((settings['grad_step']-1)*max(m_para[p],1e-20)), m_para[p] - settings['learn_rate']*(param_score - current_score)/((settings['grad_step']-1)*max(m_para[p],1e-20)))
            #print(new_m_para[p], learn*(param_score - current_score)/((settings['grad_step']-1)*test_m_para[p]), param_score-current_score)
            #print('dell',(param_score - current_score)/((settings['grad_step']-1)*max(m_para[p],1e-20)))
            if m_para[p]!=0:    
                new_m_para[p] = new_m_para[p] - learn*(param_score - current_score)/((settings['grad_step']-1)*test_m_para[p])
            else:
                new_m_para[p] = new_m_para[p] - learn*(param_score - current_score)/(0.001)
            #new_m_para[p] = new_m_para[p] - learn*(param_score - current_score)/(0.0000001)
            #if new_m_para[p]<0:
            #    new_m_para[p]=0
            #else:
                #new_m_para[p] = 0
                                                                       # d Function                   d Para = Para*(factor - 1)
                                                                                # here the factor number >1 (eg 1.001) that is multiplied by the parameter to take a step
            #if m_para[p]<min_para[p]:
             #   m_para[p] = deepcopy(min_para[p])
            #if m_para[p]>max_para[p]:
            #    m_para[p] = deepcopy(max_para[p])
            if new_m_para[p]<0:
                new_m_para[p] = 0
            if new_m_para[p]>100:
                new_m_para[p] = 100
            #x6 = time.time()
            #T2 += x3-x2
            #T3 += x4-x3
            #T4 += x5-x4
            #T5 += x6-x5
        '''if steps==settings['steps']-2:
            new_sum_m_para = sum(new_m_para.values())
            for pp in new_m_para:
                new_m_para[pp]*= new_sum_m_para/m_para_sum'''
        del m_para
        m_para = new_m_para
        del new_m_para
        if steps>0:
            if score_log[-1]<score_log[-2] and score_log[-2]/score_log[-1]<1.01:
                learn = learn*1.05
                
            elif score_log[-1]>score_log[-2]:
                learn = learn*0.8
            elif score_log[-1]==score_log[-2]:
                break
        #if score_log[-1]==max(score_log):
         #   best_mech = deepcopy(working_mech)
        #print('learn', learn)
            #if m_para[p]>max_para[p]:
             #   m_para[p] = deepcopy(max_para[p])
        #print('T1',T1)
        #print('T2',T2)
        #print('T3',T3)
        #print('T4',T4)
        #print('T5',T5)
        
    return working_mech


def get_cycle_score(data, reft_data, data_map):
    #map is ref data value, data values + weight, type
    score = 0
    count = 0
    count1 = 0
    count2 = 0
    maxy_test = sum(data[0].values())
    maxy_ref = sum(reft_data[0].values())
    #print('MAX', maxy_test, maxy_ref)
    out_score = 0
    within_score = 0
    for i in data_map:
        #print('MAX',maxy_test, maxy_ref)
        if i[0]==0:
            test = data[i[0]][i[1]]#/max(1e-20,maxy_test)
            #print('daty', data[i[0]][i[1]], maxy_test)
        else:    
            test = data[1][i[1]]
        ref = 0
        for j in i[2]:
            if j[0] in reft_data[i[0]]:
                ref = ref + reft_data[i[0]][j[0]]*j[1]
            #print(reft_data[i[0]][j[0]]*j[1])
        if i[0]==2:
            ref=ref#/max(1e-20,maxy_ref)
            #print('refy', ref, maxy_ref)
        #print(i[0],i[1], 'test', test, 'ref', ref, 'val', abs(test-ref)/max(test+ref,1e-20))
        if test<0.00001 and ref <0.00001:
            pass
        else:
            #if i[1] in [123,124]:
            score+= abs(test-ref)/max(test+ref,1e-20)
            if i[1]==123 or i[1]==124:
                pass
                #print('123',abs(test-ref)/max(test+ref,1e-20), test, ref)
                #score+= 5*abs(test-ref)/max(test+ref,1e-20)
            if i[0]==0:
                count2+=1
                if i[1] in [123,124]:
                    out_score+=abs(test-ref)/max(test+ref,1e-20)*1
                
                else:
                    out_score+=abs(test-ref)/max(test+ref,1e-20)*1
            if i[0]==1:
                count1+=1
                within_score+=abs(test-ref)/max(test+ref,1e-20)
            #print(i, test, ref, abs(test-ref)/max(test+ref,1e-20))
        count+=1
    #print('os',out_score/max(1,count2), within_score/max(1,count1))
    return out_score/max(1,count2) + within_score/max(1,count1)