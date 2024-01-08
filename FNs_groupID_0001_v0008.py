import numpy as np
import numba
import time
from math import floor, ceil


# general functions in this cell

# define a function that enumerates the identities of every agent
# @numba.jit(nopython=True)
def population_array(population_sizef, levels_of_idf, branches_per_lvlf, percent_agents_per_lvlf):
    popid = np.zeros((population_sizef, levels_of_idf), dtype=np.int64)
    idcounts = np.zeros((levels_of_idf, ), dtype=np.int64)
    parent_size = [[population_sizef]]
    branch_multiple = 1
    for idx0000 in range(0, levels_of_idf):
        count_in_branch = 0
        correct = 1
        idx0001 = 0
        while idx0001 < branch_multiple*branches_per_lvlf[idx0000]:
            if idx0000 != 0:
                psizeA = parent_size[idx0000-1][floor(idcounts[idx0000])]*percent_agents_per_lvlf[idx0000][idx0001]
                psize = floor(parent_size[idx0000-1][floor(idcounts[idx0000])]*percent_agents_per_lvlf[idx0000][idx0001])
                if psizeA != psize and correct == 1:
                    psize = ceil(parent_size[idx0000-1][floor(idcounts[idx0000])]*percent_agents_per_lvlf[idx0000][idx0001])
                    correct = 0
            else:
                psize = population_sizef
            if idx0000 != 0 and idx0001 == 0:
                parent_size.append([psize])
            elif idx0000 != 0 and idx0001 !=0:
                parent_size[idx0000].append(psize)
                
            for idx0002 in range(0, psize):
                popid[count_in_branch][idx0000] = idx0001
                count_in_branch += 1
                
            if np.sum(parent_size[idx0000]) == np.cumsum(parent_size[idx0000-1])[floor(idcounts[idx0000])]:
                idcounts[idx0000]+=1
                correct = 1
                if idcounts[idx0000] == len(parent_size[idx0000-1]) and idx0000 != 0: #correcting for small populations where niche IDs can be zero size
                    idx0001+=10000000000
            idx0001 += 1
        #stuff that needs to happen at the end of the loop:
        branch_multiple = branch_multiple*branches_per_lvlf[idx0000]
    return parent_size, popid


# function generating weights for probability of a pairing between agents
@numba.jit(nopython=True)
def connections_init(population_sizef, popidf, base_connection_weightsf, homophily_factorf):
    conweightsf = np.zeros((population_sizef, population_sizef), dtype=np.int64)
    for idx1000 in numba.prange(0, population_sizef):
        for idx1001 in numba.prange(1, population_sizef):
            idx1001b = (idx1000+idx1001)%population_sizef
            dist = np.sum(popidf[idx1000] == popidf[idx1001b])
            conweightsf[idx1000][idx1001b] = dist**homophily_factorf
    return conweightsf


#function to generate all of the random values
@numba.jit(nopython=True)
def randoms(population_sizef, epsilonf, rngf):
    randperm01f = rngf.permutation(population_sizef)
    randunif01f = rngf.random(population_sizef//2)+epsilonf
    randunif02f = rngf.random(population_sizef)
    
    return randperm01f, randunif01f, randunif02f

#function to take conweights, a permutation of pop size and random uniform of pop size and return our weighted random pairings
@numba.jit(nopython=True)
def pairings(randperm01f, randunif01f, population_sizef, conweightsf):
    halfpop = population_sizef//2
    pairsf = np.zeros((halfpop, 2), dtype=np.int64)
    conf = conweightsf.copy()
    idx2000 = 0
    count2000 = 0
    while idx2000 < halfpop:
        idx2001 = randperm01f[count2000]
        agweight = (conf[idx2001]).copy()
        agsum = np.sum(agweight)
        if agsum != 0:
            agpick = agsum*randunif01f[idx2000]
            agweight = np.cumsum(agweight)
            agweight[agweight > agpick] = -1
            agweight[agweight != -1] = 1
            agweight[agweight == -1] = 0
            pairedag = floor(np.sum(agweight))
            pairsf[idx2000] = [idx2001, pairedag]
            conf[idx2001] = 0
            conf[:, idx2001] = 0
            conf[pairedag] = 0
            conf[:, pairedag] = 0
            
            idx2000 += 1
        count2000 += 1
    
    return pairsf


# function that takes popid, population_size, levels_of_id, and branches_per_lvl; 
# then returns an array with the levels_of_id-many signals that each agent can transmit/play
# this is the binding of signal to action that will be separated in future models
# NOTE: if code is slow for large populations, may want to check whether it is faster
#       to recompute these values on each step rather than having to constantly pull a large array
#       from memory. However, since this computation depends on the popid values, I doubt it will
#       improve performance without also computing popid on die as well
@numba.jit(nopython=True)
def id_sigs_fn(popidf, population_sizef, levels_of_idf, branches_per_lvlf):
    idsigsf = popidf.copy()
    count3000 = branches_per_lvlf[0]
    for idx3001 in range(1, levels_of_idf):
        for idx3002 in range(0, population_sizef):
            idsigsf[idx3002][idx3001]+=count3000
        
        #stuff at end of loop
        count3000+=branches_per_lvlf[idx3001-1]*branches_per_lvlf[idx3001]
    return idsigsf

# function that takes weights and random uniform to determine draws
# it determins draws, not the signal/action. This separation is important
# because the draw will index reinforcement (and perhaps punishment in future models)
@numba.jit(nopython=True)
def raw_draws_fn(population_sizef, agsigweightsf, randunif02f):
    rawdrawsf = np.zeros(population_sizef, dtype=np.int64)
    for idx4000 in numba.prange(0, population_sizef):
        idxcumsum = np.cumsum(agsigweightsf[idx4000])
        idxrand = randunif02f[idx4000]*idxcumsum[-1]
        idxdraw = np.zeros(len(idxcumsum))
        idxdraw[idxcumsum<idxrand] = 1
        rawdrawsf[idx4000] = np.sum(idxdraw)
    return rawdrawsf

# function that takes agent signal weights, raw draws, pairings, idsigs, and reinforcement per level 
# to compute whethere there was successfull coordination and reinforces the weights if there was.
# The function then returns the new weights
@numba.jit(nopython=True)
def check_and_rein(population_sizef, agsigweightsf, rawdrawsf, pairsf, idsigsf, rein_per_lvlf):
    halfpopsize = population_sizef//2
    for idx5000 in numba.prange(0, halfpopsize):
        ag0 = pairsf[idx5000][0]
        ag1 = pairsf[idx5000][1]
        ag0sig = idsigsf[ag0][rawdrawsf[ag0]]
        ag1sig = idsigsf[ag1][rawdrawsf[ag1]]
        if ag0sig == ag1sig: #then there's been successfull coordination
            rawdraw = rawdrawsf[ag0] #on successfull coordination for this model ag0 and ag1 have to have drawn at the same level
#             if rawdraw != rawdrawsf[ag1]: #just checking for safety, this condition can be commented out for better performance
#                 print('error in draw match')
            reinamount = rein_per_lvlf[rawdraw]
            agsigweightsf[ag0][rawdraw] += reinamount
            agsigweightsf[ag1][rawdraw] += reinamount
        # could put an else condition here if we want to include punishment
    return agsigweightsf



# function that does everything that occurs in a single timestep of the simulation
@numba.jit(nopython=True)
def gid0001_single_tstep(idsigsf, rein_per_lvlf, conweightsf, agsigweightsf, popidf, levels_of_idf, population_sizef, epsilonf, rngf):
    randperm01ff, randunif01ff, randunif02ff = randoms(population_sizef, epsilonf, rngf)
    pairs01 = pairings(randperm01ff, randunif01ff, population_sizef, conweightsf)
    rawdrawsff = raw_draws_fn(population_sizef, agsigweightsf, randunif02ff)
    agsigweightsf = check_and_rein(population_sizef, agsigweightsf, rawdrawsff, pairs01, idsigsf, rein_per_lvlf)
    return agsigweightsf


# function to count up what level of identity is being signaled by each type after a simulation has been run
@numba.jit(nopython=True)
def final_count(agsigweightsf, popidf, levels_of_idf, population_sizef):
    final_f = np.zeros((popidf[-1][-1]+1, levels_of_idf), dtype=np.int64)
    agmax = np.argmax(agsigweightsf, axis=1)
    for idx9999 in range(0, population_sizef):
        final_f[popidf[idx9999][-1]][agmax[idx9999]] += 1
    return final_f


# function that does several timesteps
@numba.jit(nopython=True)
def gid0001_nsteps(idsigsf, rein_per_lvlf, conweightsf, agsigweightsf, popidf, levels_of_idf, population_sizef, epsilonf, rngf, nstepsf, runidf):
#     rngf = rgsf[runidf]
    for idxN in range(0, nstepsf):
        agsigweightsf = gid0001_single_tstep(idsigsf, rein_per_lvlf, conweightsf, agsigweightsf, popidf, levels_of_idf, population_sizef, epsilonf, rngf)
    final_f = final_count(agsigweightsf, popidf, levels_of_idf, population_sizef)
    return final_f, runidf