import networkx as nx
import numpy as np
from typing import List, Set, Tuple
from sklearn.cluster import SpectralClustering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class IntegrableSetDiscovery:
    """Base class for integrable set discovery methods."""
    
    def __init__(self, pairwise_integrability_matrix):
        """
        Initialize with pairwise integrability judgments.
        
        Args:
            pairwise_integrability_matrix: NxN binary matrix where [i,j]=1 
                                          if tuples i and j are integrable
        """
        self.matrix = pairwise_integrability_matrix
        self.n_tuples = len(pairwise_integrability_matrix)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build undirected graph from integrability matrix."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_tuples))
        
        for i in range(self.n_tuples):
            for j in range(i + 1, self.n_tuples):
                if self.matrix[i][j] == 1:
                    G.add_edge(i, j)
        
        return G
    
    def discover(self) -> List[Set[int]]:
        """
        Discover integrable sets.
        
        Returns:
            List of sets, where each set contains tuple indices
        """
        raise NotImplementedError


class BronKerboschMethod(IntegrableSetDiscovery):
    """
    Bron-Kerbosch algorithm for finding maximal cliques.
    
    From paper Section 4: "Each integrable set can be considered as a maximal 
    clique in a graph constructed from table T."
    """
    
    def discover(self) -> List[Set[int]]:
        """Find all maximal cliques using Bron-Kerbosch algorithm."""
        cliques = list(nx.find_cliques(self.graph))
        return [set(clique) for clique in cliques]
    
    def _bron_kerbosch(self, R: Set, P: Set, X: Set, cliques: List[Set]):
        """
        Recursive Bron-Kerbosch algorithm (Algorithm 1 from paper).
        
        Args:
            R: Current clique being constructed
            P: Candidate vertices
            X: Already processed vertices
            cliques: List to store found cliques
        """
        if len(P) == 0 and len(X) == 0:
            cliques.append(R.copy())
            return
        
        # Choose pivot
        pivot = self._choose_pivot(P.union(X))
        
        # For each vertex in P not adjacent to pivot
        for v in P.difference(self.graph.neighbors(pivot)):
            neighbors = set(self.graph.neighbors(v))
            self._bron_kerbosch(
                R.union({v}),
                P.intersection(neighbors),
                X.intersection(neighbors),
                cliques
            )
            P.remove(v)
            X.add(v)
    
    def _choose_pivot(self, vertices: Set) -> int:
        """Choose pivot vertex with maximum degree."""
        if not vertices:
            return None
        return max(vertices, key=lambda v: self.graph.degree(v))


class LouvainMethod(IntegrableSetDiscovery):
    """
    Louvain algorithm for community detection.
    
    From paper Section 4: "A modularity-based method that iteratively 
    optimizes the modularity to find a partition of the network."
    """
    
    def discover(self) -> List[Set[int]]:
        """Discover communities using Louvain algorithm."""
        from community import community_louvain
        
        # Find best partition
        partition = community_louvain.best_partition(self.graph)
        
        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            communities[comm_id].add(node)
        
        return list(communities.values())


class NewmanGirvanMethod(IntegrableSetDiscovery):
    """
    Newman-Girvan algorithm for hierarchical community detection.
    
    From paper Section 4: "A hierarchical clustering method that hierarchically 
    removes edges with 'high betweenness centrality' to identify communities."
    """
    
    def discover(self, k: int = None) -> List[Set[int]]:
        """
        Discover communities using edge betweenness.
        
        Args:
            k: Number of communities (if None, uses modularity to decide)
        """
        if k is None:
            # Use modularity-based approach
            communities = nx.community.girvan_newman(self.graph)
            # Take the first partition
            partition = next(communities)
        else:
            # Generate k communities
            communities = nx.community.girvan_newman(self.graph)
            for _ in range(k - 1):
                partition = next(communities)
        
        return [set(community) for community in partition]


class InfomapMethod(IntegrableSetDiscovery):
    """
    Infomap algorithm for community detection.
    
    From paper Section 4: "An information-theoretic approach that minimizes 
    the description length of a random walk path through the network."
    """
    
    def discover(self) -> List[Set[int]]:
        """Discover communities using Infomap."""
        try:
            import infomap
        except ImportError:
            raise ImportError("Please install infomap: pip install infomap")
        
        # Create Infomap instance
        im = infomap.Infomap()
        
        # Add edges
        for edge in self.graph.edges():
            im.add_link(edge[0], edge[1])
        
        # Run algorithm
        im.run()
        
        # Extract communities
        communities = {}
        for node in im.tree:
            if node.is_leaf:
                module_id = node.module_id
                if module_id not in communities:
                    communities[module_id] = set()
                communities[module_id].add(node.node_id)
        
        return list(communities.values())


class SpectralClusteringMethod(IntegrableSetDiscovery):
    """
    Spectral Clustering for community detection.
    
    From paper Section 4: "Uses eigenvectors from a graph Laplacian matrix 
    to partition nodes into clusters."
    """
    
    def discover(self, n_clusters: int = None) -> List[Set[int]]:
        """
        Discover communities using spectral clustering.
        
        Args:
            n_clusters: Number of clusters (if None, estimates automatically)
        """
        if n_clusters is None:
            # Estimate number of clusters from graph structure
            n_clusters = self._estimate_n_clusters()
        
        # Convert to adjacency matrix
        adj_matrix = nx.to_numpy_array(self.graph)
        
        # Apply spectral clustering
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        labels = sc.fit_predict(adj_matrix)
        
        # Group nodes by cluster
        communities = {}
        for node, label in enumerate(labels):
            if label not in communities:
                communities[label] = set()
            communities[label].add(node)
        
        return list(communities.values())
    
    def _estimate_n_clusters(self) -> int:
        """Estimate number of clusters from eigenvalue gap."""
        laplacian = nx.laplacian_matrix(self.graph).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        
        # Find largest gap in eigenvalues
        gaps = np.diff(eigenvalues)
        n_clusters = np.argmax(gaps) + 1
        
        return max(2, min(n_clusters, self.n_tuples // 2))


class GNNMethod(IntegrableSetDiscovery):
    """
    Graph Neural Network for community detection.
    
    From paper Section 4: "A DL-based representative learning approach that 
    aims to leverage graph neural networks to learn meaningful representations 
    of nodes based on the topographical structure and attributes."
    """
    
    def __init__(self, pairwise_integrability_matrix, node_features=None):
        """
        Initialize GNN method.
        
        Args:
            pairwise_integrability_matrix: NxN integrability matrix
            node_features: Optional node feature matrix (N x feature_dim)
        """
        super().__init__(pairwise_integrability_matrix)
        
        if node_features is None:
            # Use identity features if none provided
            self.node_features = torch.eye(self.n_tuples)
        else:
            self.node_features = torch.FloatTensor(node_features)
        
        self.model = None
    
    def discover(self, n_clusters: int = None, hidden_dim: int = 64, 
                 epochs: int = 100) -> List[Set[int]]:
        """
        Discover communities using GNN.
        
        Args:
            n_clusters: Number of communities
            hidden_dim: Hidden dimension size
            epochs: Training epochs
        """
        if n_clusters is None:
            n_clusters = self._estimate_n_clusters()
        
        # Prepare graph data
        data = self._prepare_data()
        
        # Build and train model
        self.model = CommunityGNN(
            num_features=self.node_features.shape[1],
            hidden_dim=hidden_dim,
            num_communities=n_clusters
        )
        
        self._train(data, epochs)
        
        # Get community assignments
        with torch.no_grad():
            embeddings = self.model.encode(data)
            assignments = self.model.cluster(embeddings)
        
        # Group nodes by community
        communities = {}
        for node, label in enumerate(assignments.numpy()):
            if label not in communities:
                communities[label] = set()
            communities[label].add(node)
        
        return list(communities.values())
    
    def _prepare_data(self):
        """Prepare PyTorch Geometric data object."""
        # Get edge list
        edge_list = list(self.graph.edges())
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        
        # Create bidirectional edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        return Data(x=self.node_features, edge_index=edge_index)
    
    def _train(self, data, epochs):
        """Train GNN model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            embeddings = self.model.encode(data)
            loss = self.model.loss(embeddings, data.edge_index)
            
            loss.backward()
            optimizer.step()
    
    def _estimate_n_clusters(self) -> int:
        """Estimate number of clusters."""
        # Use similar logic as spectral clustering
        return SpectralClusteringMethod(self.matrix)._estimate_n_clusters()


class CommunityGNN(nn.Module):
    """GNN model for community detection."""
    
    def __init__(self, num_features, hidden_dim, num_communities):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.cluster_layer = nn.Linear(hidden_dim, num_communities)
    
    def encode(self, data):
        """Encode nodes to embeddings."""
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return x
    
    def cluster(self, embeddings):
        """Assign nodes to clusters."""
        logits = self.cluster_layer(embeddings)
        return torch.argmax(logits, dim=1)
    
    def loss(self, embeddings, edge_index):
        """Compute contrastive loss."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Positive pairs (connected nodes)
        pos_mask = torch.zeros_like(sim_matrix)
        pos_mask[edge_index[0], edge_index[1]] = 1
        
        # Contrastive loss
        pos_sim = (sim_matrix * pos_mask).sum() / pos_mask.sum()
        neg_sim = (sim_matrix * (1 - pos_mask)).sum() / (1 - pos_mask).sum()
        
        return -pos_sim + neg_sim

