"""
Tests...

Author : Sandro Vega Pons

License:
"""
import numpy as np
import networkx as nx
from gk_random_walks import GK_RW
from gk_shortest_path import GK_SP
from gk_weisfeiler_lehman import GK_WL
from PyBDGK.IntermRepresentation.IntermRepresentation import IntermRep
from PyBDGK.GraphEncoding.GE_NeighConst_HCA import GE_NeighConst_HCA
from bct.bct import *
import pdb

g1 = nx.from_edgelist([(0,1),(1,2),(1,3),(2,4)])        
g2 = nx.from_edgelist([(0,3),(1,4),(2,1),(0,2)])


def test_rw():
    wl = GK_WL()
    sp = GK_SP()
    
    print wl.compare(g1, g2)
    print wl.compare_normalized(g1, g2)
    
    print sp.compare(g1, g2)
    print sp.compare_normalized(g1, g2)
        



def test_bct():
    """
    """
    s1 = '/home/sandrovegapons/Documents/Data/UriDataSet/xyz_coords_graymattermask_ANGO'
    f1 = '/home/sandrovegapons/Documents/Data/UriDataSet/blur.1.ANGO.steadystate.TRIM_graymask_dump'
    arr_xyz = np.genfromtxt(s1, dtype = float, delimiter = ' ')
    arr_voxels = np.genfromtxt(f1, dtype = float, delimiter = ' ') 
    int_rep = IntermRep(arr_voxels, arr_xyz)
    fc =  GE_NeighConst_HCA()
    g = fc.encode(int_rep, neighborhood_size = 26,
                                clust_ratio = 120, encoding='geometrical',
                                similarity_measure='pearson',
                                threshold=0.4, num_lab=10, n_jobs = 1) 
     
    n = nx.number_of_nodes(g)                           
    adj = np.array(nx.adjacency_matrix(g))
    print type(adj)
    print adj
    bt_bin = betweenness_bin(adj)
    print type(bt_bin[0])
    print bt_bin.shape
    bt_bin = bt_bin/((n-1)*(n-2))
    bt = np.mean(bt_bin)
    print bt_bin
    print bt
    
    
    
def complex_network_mapping(graph):
    """
    Compute the vectorial mapping of a graph based on the computation of
    several complex-network analysis indexes.
    """
    vect = []    
    
    n = nx.number_of_nodes(graph)
    e = nx.number_of_edges(graph)
    print n,e
    
    adj = np.array(nx.adjacency_matrix(graph))
    adj_bin = np.where(adj>0, 1., 0.)
    adj_conn = 1 - adj
    
    #Node Betweenness binary
    bt_bin = betweenness_bin(adj_bin)
    bt_bin = bt_bin/((n-1)*(n-2))
    avg_btb = np.mean(bt_bin)
    vect.append(avg_btb)

    #Node betweenness weighted
    bt_wei = betweenness_wei(adj_conn)
    bt_wei = bt_wei/((n-1)*(n-2))
    avg_btw = np.mean(bt_wei)
    vect.append(avg_btw)
    
    #Edge betweenness binary
    ebt_bin,_ = edge_betweenness_bin(adj_bin)
    ebt_bin = ebt_bin/((n-1)*(n-2))
    avg_ebtb = np.mean(ebt_bin)
    vect.append(avg_ebtb)
    
    #Edge betweenness weighted
    ebt_wei,_ = edge_betweenness_wei(adj_conn)
    ebt_wei = ebt_wei/((n-1)*(n-2))
    avg_ebtw = np.mean(ebt_wei)
    vect.append(avg_ebtw)
    
    #Eigen vector centrality binary    
    evc_bin = eigenvector_centrality_und(adj_bin)
    avg_evcb = np.mean(evc_bin)
    vect.append(avg_evcb)
    
    #Eigen vector centrality weighted    
    evc_bin = eigenvector_centrality_und(adj)
    avg_evcb = np.mean(evc_bin)
    vect.append(avg_evcb)
    
    #Erange
    era_bin,_,_,_ = erange(adj_bin)
    avg_era = np.mean(era_bin)
    vect.append(avg_era)
    
    #Flow coefficient    
    _,flow_bin,_ = flow_coef_bd(adj_bin)
    avg_flow = np.mean(flow_bin)
    vect.append(avg_flow)
    
    #Kcoreness centrality
    kcor_bin,_ = kcoreness_centrality_bu(adj_bin)
    avg_kcor = np.mean(kcor_bin)
    vect.append(avg_kcor)
    
    #Page rank centrality
    pgr_wei = pagerank_centrality(adj,d=0.85)
    avg_pgr = np.mean(pgr_wei)
    vect.append(avg_pgr)
    
    #Subgraph centrality
#    sgc_bin = subgraph_centrality(adj_bin)
#    avg_sgc = np.mean(sgc_bin)
#    vect.append(avg_sgc)
    
    return vect
    
    
  
    
    


    
    
if __name__ == "__main__":
#    test_rw()   
#    test_bct()      

    s1 = '/home/sandrovegapons/Documents/Data/UriDataSet/xyz_coords_graymattermask_ANGO'
    f1 = '/home/sandrovegapons/Documents/Data/UriDataSet/blur.1.ANGO.steadystate.TRIM_graymask_dump'
    arr_xyz = np.genfromtxt(s1, dtype = float, delimiter = ' ')
    arr_voxels = np.genfromtxt(f1, dtype = float, delimiter = ' ') 
    int_rep = IntermRep(arr_voxels, arr_xyz)
    fc =  GE_NeighConst_HCA()
    g = fc.encode(int_rep, neighborhood_size = 26,
                                clust_ratio = 120, encoding='geometrical',
                                similarity_measure='pearson',
                                threshold=0.25, num_lab=10, n_jobs = 1)  
                                
    complex_network_mapping(g)
    
        
        
        
        
        
        
        
        
    
        