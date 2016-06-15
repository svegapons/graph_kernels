import numpy as np
from gk_base import GK_Base
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
import pdb


class GK_DRE(GK_Base):
    """
    Graph kernel based on Direct Connection Embedding (DCE) + a kernel for
    vector data (sklearn.metrics.pairwise.pairwise_kernels are used).  
    Only works with data that hold node-correspondence!
    """       
    
    def __init__(self, th=0., val=1, kernel_vector_space='rbf', **kwds):
        """
        Parameters
        ----------
        th: float
            Threshold to edge weights
        K : int
            Value 
        kernel_vector_space: string 
            It must be one of the metrics in sklearn 
            pairwise.PAIRWISE_KERNEL_FUNCTIONS. i.e. one of:
            ['rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']
        **kwds : optional keyword parameters
            Additional keyword arguments for the kernel_vector_space function.
        """
        self.th = th
        self.val = val
        self.kernel = kernel_vector_space
        self.kwds = kwds
    
        
    def compare_pairwise(self, graph_list, n_jobs=1, normalize=True):
        """Compute the all-pairs kernel values for a list of graphs.       
        
        Parameters
        ----------
        graph_list: list of ndarray
                    A list of graphs (adjacency matrices)
        
        Return
        ------
        K: ndarray, shape = (len(graph_list), len(graph_list))
            The similarity matrix of all graphs in graph_list.
        """
        n_graphs = len(graph_list)
        #Indices of upper triangular part of the matrix. Same indices for 
        #all graphs since we are assuming node correspondence
        id_x, id_y = np.triu_indices_from(graph_list[0], k=1)
        n_feats = len(id_x)
        X = np.zeros((n_graphs, n_feats))
        
        #Unfolding upper adjacency matrices
        for i, g in enumerate(graph_list):
            X[i] = g[id_x, id_y]
            
        
        #Thresholding the vector embedding and changing from similarities to 
        #dissimilarities.
        X = np.where(X > self.th, np.max(X) - X, 0)
#        X = np.where(X > self.th, X, 0)
        
        XX = np.zeros((X.shape[0], X.shape[0]))
        for t,v in enumerate(X):
            for q,u in enumerate(X):
                aux = 0
                for i in range(len(v)):
                    if v[i]==0 or u[i]==0:
                        aux += self.val
                    else:
                        aux += np.abs(u[i] - v[i])
                XX[t,q] = aux   
                                     
        #Heuristic for gamma..
        if self.kernel == 'rbf':
            if not 'gamma' in self.kwds:
                sigma2 = np.median(pairwise_distances(XX, metric='euclidean'))**2
                self.kwds['gamma'] = 1./sigma2
                
        #Applying the pairwise kernel       
        K = pairwise_kernels(XX, XX, metric=self.kernel, n_jobs=n_jobs, 
                             **self.kwds)
        
        if normalize:
            K = self._normalize(K)
            
        return K
    
    
    
    
    
    
        
        