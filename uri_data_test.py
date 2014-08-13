import numpy as np
import networkx as nx
import os
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, KFold
from PyBDGK.IntermRepresentation.IntermRepresentation import IntermRep
from PyBDGK.GraphEncoding.GE_NeighConst_HCA import GE_NeighConst_HCA
from gk_shortest_path import GK_SP
from gk_weisfeiler_lehman import GK_WL
from sklearn.preprocessing import StandardScaler
from bct.bct import *
import pdb
import logging
    

def gk_comparison_uri_data(directory):
    """
    Experiment with the Uri dataset.
    Comparison of different graph kernels for classification

    Parameters
    ----------
    directory: string
        The path of the directory containing all data files.
    """
     #Loading all files in the folder
    xyz_files = []
    blur_files = []

    #Spliting files in xyz coordinates and voxels data.
    dirs = os.listdir(directory)
    dirs.sort()
    for f in dirs:
        if os.path.isfile(os.path.join(directory, f)):
            if f.startswith('xyz'):
                xyz_files.append(f)
            if f.startswith('blur'):
                blur_files.append(f)

    #Loading xyz data
    dict_xyz = {}
    for f in xyz_files:
        #The name of the subject is given by the four last letter.
        dict_xyz[f[-4:]] = np.genfromtxt(os.path.join(directory, f),
                                         dtype = float, delimiter = ' ')
        print "xyz_file for subject %s was loaded." %(f[-4:])

    #Loading voxels data and creating the intermediate representation objects
    inter_reps = []
    for f in blur_files:
        #Name of the subject is always in positions [7:11]
        s_name = f[7:11]
        #Class is in possition 5
        cls = int(f[5])
        arr_voxels = np.genfromtxt(os.path.join(directory, f), dtype = float,
                                       delimiter = ' ')
        inter_reps.append(IntermRep(arr_voxels, dict_xyz[s_name], s_name,
                                       cls))

        print "Intermediate representation for subject %s and class %s created." %(s_name, cls)

    #Computing the graph encoding
    graphs = []
    classes = []
    subjects = []
    
    # Graph encoding based on Neirghboring connections and hierarchical clustering algorithm.
    fc =  GE_NeighConst_HCA()
    for i_rep in inter_reps:
        graphs.append(fc.encode(i_rep, neighborhood_size = 26,
                                clust_ratio = 120, encoding='geometrical',
                                similarity_measure='pearson',
                                threshold=0.6, num_lab=10, n_jobs = 1))
        classes.append(i_rep.cls)
        subjects.append(i_rep.subj_name)
        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
                                                           graphs[-1].number_of_edges())
        print ""
    
    #Reordering data for the leave-one-subject-out cross-validation
    nm_graphs = [None] * len(graphs)
    nm_classes = [None] * len(classes)
    nm_subjects = [None] * len(subjects)

    for i in range(len(graphs) / 2):
        nm_graphs[i*2] = graphs[i]
        nm_graphs[i*2 + 1] = graphs[(len(graphs) / 2) + i]
        nm_classes[i*2] = classes[i]
        nm_classes[i*2 + 1] = classes[(len(classes) / 2) + i]
        nm_subjects[i*2] = subjects[i]
        nm_subjects[i*2 + 1] = subjects[(len(subjects) / 2) + i]

    print nm_subjects
    print nm_classes
    
    
    ############
    #logging
    logger = logging.getLogger('graph_kernel_comparison')    
    logger.level = logging.DEBUG    
    stderr_log_handler = logging.StreamHandler()
    logger.addHandler(stderr_log_handler)    
    # nice output format
    formatter = logging.Formatter('%(asctime)s- %(message)s')
    stderr_log_handler.setFormatter(formatter)
    #################################    
    
    #Computing the kernel matrix by using WL graph kernel.
    gk_wl = GK_WL()
    wl_mat = gk_wl.compare_list_normalized(nm_graphs, h = 1, nl = False) 

    #Computing the kernel matrix by using SP graph kernel
    gk_sp = GK_SP()
    sp_mat = gk_sp.compare_list_normalized(nm_graphs) 
#    pdb.set_trace()
    
    #Leave-one-subject-out cross-validation for shortest-path graph kernel
    print "Running leave-one-subject-out cross-validation for Shortest-Path kernel"
    sp_mean_score = cv_precomputed_matrix(sp_mat, np.array(nm_classes),
                                       n_subjects = 19, logger=logger)
    print "Mean score for SP: %s" %(sp_mean_score)
    
    
    #Leave-one-subject-out cross-validation for weisfeiler-lehman graph kernel
    print "Running leave-one-subject-out cross-validation fro Weisfeiler-Lehman kernel"
    wl_mean_score = cv_precomputed_matrix(wl_mat, np.array(nm_classes),
                                       n_subjects = 19, logger=logger)
    print "Mean score for WL: %s" %(wl_mean_score)
    

                                                  


def cv_precomputed_matrix(kernel_matrix, labels, n_subjects, logger):
    """
    Cross-validation with SVM and precomputed kernel.
    """
    clf = SVC(kernel = 'precomputed')
    cv_scores = cross_val_score(clf, kernel_matrix, labels, cv=KFold(len(labels), n_subjects, shuffle=False))
    #cv_scores = cross_val_score(clf, kernel_matrix, labels, cv=n_subjects)

#    print "Scores: %s" %(cv_scores)
    logger.info("    Mean accuracy: %s" %(np.mean(cv_scores)))
    logger.info("    Standard Deviation: %s" %(np.std(cv_scores)))
    logger.info("    ---------------------------------------------------------------")
    
    return np.mean(cv_scores)




        
def complex_networks_mapping_uri_data(directory):
    """

    Parameters
    ----------
    directory: string
        The path of the directory containing all data files.
    """
     #Loading all files in the folder
    xyz_files = []
    blur_files = []

    #Spliting files in xyz coordinates and voxels data.
    dirs = os.listdir(directory)
    dirs.sort()
    for f in dirs:
        if os.path.isfile(os.path.join(directory, f)):
            if f.startswith('xyz'):
                xyz_files.append(f)
            if f.startswith('blur'):
                blur_files.append(f)

    #Loading xyz data
    dict_xyz = {}
    for f in xyz_files:
        #The name of the subject is given by the four last letter.
        dict_xyz[f[-4:]] = np.genfromtxt(os.path.join(directory, f),
                                         dtype = float, delimiter = ' ')
        print "xyz_file for subject %s was loaded." %(f[-4:])

    #Loading voxels data and creating the intermediate representation objects
    inter_reps = []
    for f in blur_files:
        #Name of the subject is always in positions [7:11]
        s_name = f[7:11]
        #Class is in possition 5
        cls = int(f[5])
        arr_voxels = np.genfromtxt(os.path.join(directory, f), dtype = float,
                                       delimiter = ' ')
        inter_reps.append(IntermRep(arr_voxels, dict_xyz[s_name], s_name,
                                       cls))

        print "Intermediate representation for subject %s and class %s created." %(s_name, cls)

    #Computing the graph encoding
    graphs = []
    classes = []
    subjects = []
    vects = []
    
    # Graph encoding based on Neirghboring connections and hierarchical clustering algorithm.
    fc =  GE_NeighConst_HCA()
    for i_rep in inter_reps:
        graphs.append(fc.encode(i_rep, neighborhood_size = 26,
                                clust_ratio = 60, encoding='geometrical',
                                similarity_measure='pearson',
                                threshold=0.25, num_lab=10, n_jobs = 1))
        classes.append(i_rep.cls)
        subjects.append(i_rep.subj_name)
        vects.append(complex_network_mapping(graphs[-1]))
        print "Graph built for subject %s and class %s." %(i_rep.subj_name, i_rep.cls)
        print "Number of nodes: %i, number of edges: %i" %(graphs[-1].number_of_nodes(),
                                                           graphs[-1].number_of_edges())
        print ""
    
    #Reordering data for the leave-one-subject-out cross-validation
    nm_graphs = [None] * len(graphs)
    nm_classes = [None] * len(classes)
    nm_subjects = [None] * len(subjects)
    nm_vects = [None] * len(vects)

    for i in range(len(graphs) / 2):
        nm_graphs[i*2] = graphs[i]
        nm_graphs[i*2 + 1] = graphs[(len(graphs) / 2) + i]
        nm_classes[i*2] = classes[i]
        nm_classes[i*2 + 1] = classes[(len(classes) / 2) + i]
        nm_subjects[i*2] = subjects[i]
        nm_subjects[i*2 + 1] = subjects[(len(subjects) / 2) + i]
        nm_vects[i*2] = vects[i]
        nm_vects[i*2 + 1] = vects[(len(vects) / 2) + i]

    print nm_subjects
    print nm_classes
    
    nm_vects = np.array(nm_vects)
    nm_vects = np.where(nm_vects==inf, 10, nm_vects)
    nm_vects = np.where(nm_vects==nan, 10, nm_vects)
    
    ss = StandardScaler() 
    X = ss.fit_transform(nm_vects)
    print X
    print np.mean(X)
    print np.max(X)
    
    clf = SVC()
    cv_scores = cross_val_score(clf, X, np.array(nm_classes), cv=KFold(len(nm_classes), 19, shuffle=False))   
    print cv_scores  
    print np.mean(cv_scores)
        


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
    
#    #Edge betweenness binary
#    ebt_bin,_ = edge_betweenness_bin(adj_bin)
#    ebt_bin = ebt_bin/((n-1)*(n-2))
#    avg_ebtb = np.mean(ebt_bin)
#    vect.append(avg_ebtb)
#    
#    #Edge betweenness weighted
#    ebt_wei,_ = edge_betweenness_wei(adj_conn)
#    ebt_wei = ebt_wei/((n-1)*(n-2))
#    avg_ebtw = np.mean(ebt_wei)
#    vect.append(avg_ebtw)
    
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
        
        
        