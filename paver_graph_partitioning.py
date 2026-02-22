#!/usr/bin/env python3
"""
PAVER Graph Partitioning Module

This module implements the three graph partitioning algorithms from the PAVER paper:
1. MST-TS  - Maximum Spanning Tree based TB Scheduler
2. K-Way-TS - K-Way Partition based TB Scheduler  
3. RB-TS   - Recursive Bi-Partitioning based TB Scheduler

These algorithms partition a locality graph (from LocalityGuru) to maximize
cache locality when scheduling Thread Blocks (TBs) to Streaming Multiprocessors (SMs).

Reference:
    D. Tripathy et al., "PAVER: Locality Graph-Based Thread Block Scheduling for GPUs"
    ACM Transactions on Architecture and Code Optimization, Vol. 18, No. 3, 2021
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import heapq
import copy


class PartitioningMethod(Enum):
    """Enumeration of available partitioning methods"""
    MST_TS = "mst_ts"
    KWAY_TS = "kway_ts"
    RB_TS = "rb_ts"


@dataclass
class LocalityGraph:
    """
    Represents a locality graph from LocalityGuru analysis.
    
    The locality graph captures inter-TB data sharing where:
    - Vertices represent Thread Blocks (TBs)
    - Edge weights represent the number of shared data references between TBs
    
    Attributes:
        num_tbs: Total number of thread blocks
        adjacency_matrix: NxN matrix where element [i,j] = shared data refs between TB_i and TB_j
        tb_ids: List of TB identifiers (default: 0 to num_tbs-1)
    """
    num_tbs: int
    adjacency_matrix: np.ndarray
    tb_ids: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.tb_ids) == 0:
            self.tb_ids = list(range(self.num_tbs))
        
        # Ensure adjacency matrix is symmetric (undirected graph)
        if not np.allclose(self.adjacency_matrix, self.adjacency_matrix.T):
            self.adjacency_matrix = (self.adjacency_matrix + self.adjacency_matrix.T) / 2
    
    def get_edge_weight(self, tb_i: int, tb_j: int) -> float:
        """Get the edge weight (shared data refs) between two TBs"""
        return self.adjacency_matrix[tb_i, tb_j]
    
    def get_neighbors(self, tb: int) -> List[Tuple[int, float]]:
        """Get all neighbors of a TB with their edge weights"""
        neighbors = []
        for j in range(self.num_tbs):
            if j != tb and self.adjacency_matrix[tb, j] > 0:
                neighbors.append((j, self.adjacency_matrix[tb, j]))
        return neighbors
    
    def get_total_edge_weight(self) -> float:
        """Get sum of all edge weights in the graph"""
        return np.sum(self.adjacency_matrix) / 2  # Divide by 2 since matrix is symmetric
    
    def get_subgraph(self, tb_subset: List[int]) -> 'LocalityGraph':
        """Extract a subgraph containing only the specified TBs"""
        n = len(tb_subset)
        sub_matrix = np.zeros((n, n))
        for i, tb_i in enumerate(tb_subset):
            for j, tb_j in enumerate(tb_subset):
                sub_matrix[i, j] = self.adjacency_matrix[tb_i, tb_j]
        return LocalityGraph(n, sub_matrix, tb_subset)


@dataclass
class GPUConfig:
    """
    GPU hardware configuration parameters.
    
    Attributes:
        num_sms: Number of Streaming Multiprocessors
        max_tb_per_sm: Maximum concurrent TBs per SM (determined by resources)
        l1_cache_size_kb: L1 cache size in KB
        l2_cache_size_kb: L2 cache size in KB
    """
    num_sms: int
    max_tb_per_sm: int
    l1_cache_size_kb: int = 48
    l2_cache_size_kb: int = 1536
    
    def get_total_sm_capacity(self) -> int:
        """Get total TB capacity across all SMs"""
        return self.num_sms * self.max_tb_per_sm


@dataclass
class TBGroup:
    """
    A group of Thread Blocks to be scheduled together.
    
    Attributes:
        tb_list: Ordered list of TB IDs in this group
        partition_id: Identifier for the partition this group belongs to
        total_locality: Sum of edge weights within this group
    """
    tb_list: List[int]
    partition_id: int = 0
    total_locality: float = 0.0
    
    def __len__(self):
        return len(self.tb_list)
    
    def __iter__(self):
        return iter(self.tb_list)


@dataclass
class PartitionResult:
    """
    Result of graph partitioning.
    
    Attributes:
        method: The partitioning method used
        partitions: List of TBGroup objects representing partitions
        tb_schedule: Flattened ordered list of TB IDs for scheduling
        partition_assignment: Dict mapping TB ID to partition ID
        total_intra_partition_locality: Sum of edge weights within partitions
        total_inter_partition_cut: Sum of edge weights cut between partitions
        sm_assignments: Dict mapping SM ID to list of TBGroups
    """
    method: PartitioningMethod
    partitions: List[TBGroup]
    tb_schedule: List[int]
    partition_assignment: Dict[int, int]
    total_intra_partition_locality: float
    total_inter_partition_cut: float
    sm_assignments: Dict[int, List[TBGroup]] = field(default_factory=dict)


# ==============================================================================
# MST-TS: Maximum Spanning Tree based TB Scheduler
# ==============================================================================

class PrimMST:
    """
    Prim's algorithm for Maximum Spanning Tree construction.
    
    Used to find a path through the locality graph that maximizes
    total edge weight (data sharing) along the path.
    """
    
    @staticmethod
    def compute_mst(graph: LocalityGraph) -> List[Tuple[int, int, float]]:
        """
        Compute Maximum Spanning Tree using Prim's algorithm.
        
        Args:
            graph: LocalityGraph to compute MST for
            
        Returns:
            List of edges (tb_i, tb_j, weight) in the MST
        """
        n = graph.num_tbs
        if n == 0:
            return []
        
        # Track visited nodes
        visited = [False] * n
        mst_edges = []
        
        # Priority queue: (-weight, from_tb, to_tb)
        # Negative weight for max-heap behavior
        pq = []
        
        # Start from TB 0
        visited[0] = True
        
        # Add all edges from TB 0 to priority queue
        for neighbor, weight in graph.get_neighbors(0):
            heapq.heappush(pq, (-weight, 0, neighbor))
        
        while pq and len(mst_edges) < n - 1:
            neg_weight, from_tb, to_tb = heapq.heappop(pq)
            
            if visited[to_tb]:
                continue
                
            visited[to_tb] = True
            mst_edges.append((from_tb, to_tb, -neg_weight))
            
            # Add edges from newly visited node
            for neighbor, weight in graph.get_neighbors(to_tb):
                if not visited[neighbor]:
                    heapq.heappush(pq, (-weight, to_tb, neighbor))
        
        return mst_edges
    
    @staticmethod
    def get_mst_ordering(graph: LocalityGraph) -> List[int]:
        """
        Get TB ordering based on MST traversal (DFS order).
        
        This creates an ordering where consecutive TBs in the list
        have maximum data sharing along the MST path.
        
        Args:
            graph: LocalityGraph to order
            
        Returns:
            Ordered list of TB IDs
        """
        n = graph.num_tbs
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        mst_edges = PrimMST.compute_mst(graph)
        
        # Build adjacency list from MST edges
        adj = {i: [] for i in range(n)}
        for u, v, w in mst_edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # DFS traversal to get ordering
        visited = [False] * n
        ordering = []
        stack = []
        
        def dfs_iterative(start):
            """Iterative DFS to avoid recursion limit"""
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                ordering.append(node)
                # Sort neighbors by weight (ascending) so highest weight is popped first
                neighbors = sorted(adj[node], key=lambda x: x[1])
                for neighbor, _ in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)
        
        # Start DFS from node with highest degree in MST
        start_node = max(range(n), key=lambda x: len(adj[x])) if adj else 0
        dfs_iterative(start_node)
        
        # Handle disconnected components
        for i in range(n):
            if not visited[i]:
                dfs_iterative(i)
        
        return ordering


class MST_TS:
    """
    Maximum Spanning Tree based Thread Block Scheduler (MST-TS).
    
    This scheduler uses Prim's MST to order TBs such that consecutive TBs
    have maximum data sharing. TBs are then partitioned into groups of size x
    (max concurrent TBs per SM) and assigned to SMs in round-robin fashion.
    
    Algorithm:
    1. Construct MST of locality graph using Prim's algorithm
    2. Order TBs by traversing the MST
    3. Partition consecutive TBs into groups of size x (max TBs per SM)
    4. Assign first x TBs as initial group to SM, then subsequent TBs one at a time
    """
    
    def __init__(self, graph: LocalityGraph, gpu_config: GPUConfig):
        self.graph = graph
        self.config = gpu_config
    
    def partition(self) -> PartitionResult:
        """
        Perform MST-based partitioning.
        
        Returns:
            PartitionResult containing the TB schedule and assignments
        """
        # Step 1: Get MST ordering
        mst_ordering = PrimMST.get_mst_ordering(self.graph)
        
        # Step 2: Create TB groups
        # First group for each SM has x TBs, subsequent groups have 1 TB each
        x = self.config.max_tb_per_sm
        n_sms = self.config.num_sms
        n_tbs = len(mst_ordering)
        
        partitions = []
        partition_assignment = {}
        sm_assignments = {sm: [] for sm in range(n_sms)}
        
        # Initial assignment: First (n_sms * x) TBs go as initial groups
        initial_tbs = min(n_sms * x, n_tbs)
        
        current_idx = 0
        partition_id = 0
        
        # Create initial groups of size x for each SM
        for sm in range(n_sms):
            if current_idx >= n_tbs:
                break
            
            # Take up to x TBs for initial group
            group_end = min(current_idx + x, n_tbs)
            group_tbs = mst_ordering[current_idx:group_end]
            
            if group_tbs:
                locality = self._compute_group_locality(group_tbs)
                group = TBGroup(group_tbs, partition_id, locality)
                partitions.append(group)
                sm_assignments[sm].append(group)
                
                for tb in group_tbs:
                    partition_assignment[tb] = partition_id
                
                partition_id += 1
            current_idx = group_end
        
        # Remaining TBs: assigned one at a time in round-robin to SMs
        sm_idx = 0
        while current_idx < n_tbs:
            tb = mst_ordering[current_idx]
            group = TBGroup([tb], partition_id, 0.0)
            partitions.append(group)
            sm_assignments[sm_idx].append(group)
            partition_assignment[tb] = partition_id
            
            partition_id += 1
            sm_idx = (sm_idx + 1) % n_sms
            current_idx += 1
        
        # Compute locality metrics
        intra_locality = sum(p.total_locality for p in partitions)
        total_locality = self.graph.get_total_edge_weight()
        inter_cut = total_locality - intra_locality
        
        return PartitionResult(
            method=PartitioningMethod.MST_TS,
            partitions=partitions,
            tb_schedule=mst_ordering,
            partition_assignment=partition_assignment,
            total_intra_partition_locality=intra_locality,
            total_inter_partition_cut=inter_cut,
            sm_assignments=sm_assignments
        )
    
    def _compute_group_locality(self, tb_list: List[int]) -> float:
        """Compute sum of edge weights within a TB group"""
        locality = 0.0
        for i, tb_i in enumerate(tb_list):
            for tb_j in tb_list[i+1:]:
                locality += self.graph.get_edge_weight(tb_i, tb_j)
        return locality


# ==============================================================================
# K-Way-TS: K-Way Partition based TB Scheduler
# ==============================================================================

class METIS:
    """
    METIS-style graph partitioning implementation.
    
    This is a simplified implementation of multi-level graph partitioning
    that mimics METIS behavior. For production use, consider using the
    actual pymetis library.
    
    The algorithm aims to:
    1. Minimize edge cut (edges crossing partition boundaries)
    2. Balance partition sizes
    3. Maximize intra-partition edge weight sum
    """
    
    @staticmethod
    def partition_graph(graph: LocalityGraph, n_partitions: int, 
                       balance_factor: float = 1.05) -> List[List[int]]:
        """
        Partition the graph into n_partitions balanced parts.
        
        Uses a multi-level approach with coarsening, initial partitioning,
        and refinement phases.
        
        Args:
            graph: LocalityGraph to partition
            n_partitions: Number of partitions (k)
            balance_factor: Maximum imbalance ratio (1.05 = 5% imbalance allowed)
            
        Returns:
            List of TB lists, one per partition
        """
        n = graph.num_tbs
        
        if n <= n_partitions:
            # Edge case: only 1 TB
            if n <= 1:
                return [[i for i in range(n)]]
            
            # Reduce partition count to ensure meaningful partitions
            # Goal: at least 2 TB
            n_partitions = max(1, n // 2)
            
            # If after adjustment we have more TBs than partitions, 
            # fall through to normal algorithm
            if n > n_partitions:
                pass  # Continue to coarsening below
            else:
                # Simple balanced distribution for very small graphs
                partitions = [[] for _ in range(n_partitions)]
                for i in range(n):
                    partitions[i % n_partitions].append(i)
                return partitions
            # Each TB gets its own partition
            # return [[i] for i in range(n)]
        
        # Phase 1: Coarsen the graph
        coarsened, mapping = METIS._coarsen(graph, n_partitions * 4)
        
        # Phase 2: Initial partitioning on coarsened graph
        initial_parts = METIS._initial_partition(coarsened, n_partitions)
        
        # Phase 3: Uncoarsen and refine
        partitions = METIS._uncoarsen_and_refine(
            graph, coarsened, initial_parts, mapping, n_partitions, balance_factor
        )
        
        return partitions
    
    @staticmethod
    def bi_partition(graph: LocalityGraph, balance_factor: float = 1.05) -> Tuple[List[int], List[int]]:
        """
        Partition graph into exactly 2 balanced parts.
        
        Used by recursive bi-partitioning (RB-TS).
        
        Args:
            graph: LocalityGraph to partition
            balance_factor: Maximum imbalance ratio
            
        Returns:
            Tuple of two TB lists
        """
        result = METIS.partition_graph(graph, 2, balance_factor)
        if len(result) == 1:
            # Split evenly if couldn't partition
            mid = len(result[0]) // 2
            return result[0][:mid], result[0][mid:]
        return result[0], result[1]
    
    @staticmethod
    def _coarsen(graph: LocalityGraph, target_size: int) -> Tuple[LocalityGraph, Dict[int, List[int]]]:
        """
        Coarsen graph by collapsing heavy edges.
        
        Returns:
            Tuple of (coarsened graph, mapping from coarse to fine nodes)
        """
        n = graph.num_tbs
        
        if n <= target_size:
            # No coarsening needed
            identity_map = {i: [i] for i in range(n)}
            return graph, identity_map
        
        # Heavy edge matching
        matched = [False] * n
        matches = []  # List of (u, v) pairs to merge
        
        # Sort edges by weight (descending)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                w = graph.adjacency_matrix[i, j]
                if w > 0:
                    edges.append((w, i, j))
        edges.sort(reverse=True)
        
        # Greedily match along heavy edges
        for w, u, v in edges:
            if not matched[u] and not matched[v]:
                matches.append((u, v))
                matched[u] = True
                matched[v] = True
        
        # Create coarse nodes
        coarse_id = 0
        fine_to_coarse = {}
        coarse_to_fine = {}
        
        for u, v in matches:
            fine_to_coarse[u] = coarse_id
            fine_to_coarse[v] = coarse_id
            coarse_to_fine[coarse_id] = [u, v]
            coarse_id += 1
        
        # Unmatched nodes become singleton coarse nodes
        for i in range(n):
            if not matched[i]:
                fine_to_coarse[i] = coarse_id
                coarse_to_fine[coarse_id] = [i]
                coarse_id += 1
        
        # Build coarsened adjacency matrix
        n_coarse = coarse_id
        coarse_adj = np.zeros((n_coarse, n_coarse))
        
        for i in range(n):
            for j in range(i + 1, n):
                ci = fine_to_coarse[i]
                cj = fine_to_coarse[j]
                if ci != cj:
                    coarse_adj[ci, cj] += graph.adjacency_matrix[i, j]
                    coarse_adj[cj, ci] = coarse_adj[ci, cj]
        
        coarse_graph = LocalityGraph(n_coarse, coarse_adj)
        
        # Recursively coarsen if still too large
        if n_coarse > target_size:
            deeper_graph, deeper_map = METIS._coarsen(coarse_graph, target_size)
            # Compose mappings
            composed_map = {}
            for coarse_node, fine_nodes in coarse_to_fine.items():
                deeper_coarse = list(deeper_map.keys())[0]  # Find corresponding
                for dc, fc_list in deeper_map.items():
                    if coarse_node in fc_list:
                        deeper_coarse = dc
                        break
                if deeper_coarse not in composed_map:
                    composed_map[deeper_coarse] = []
                composed_map[deeper_coarse].extend(fine_nodes)
            return deeper_graph, composed_map
        
        return coarse_graph, coarse_to_fine
    
    @staticmethod
    def _initial_partition(graph: LocalityGraph, n_partitions: int) -> List[List[int]]:
        """
        Create initial partition using greedy algorithm.
        """
        n = graph.num_tbs
        partitions = [[] for _ in range(n_partitions)]
        target_size = n // n_partitions
        
        assigned = [False] * n
        
        # Seed each partition with highest connectivity node
        # that hasn't been assigned
        for p in range(n_partitions):
            if all(assigned):
                break
                
            # Find unassigned node with highest total weight
            best_node = -1
            best_weight = -1
            for i in range(n):
                if not assigned[i]:
                    total_weight = sum(graph.adjacency_matrix[i])
                    if total_weight > best_weight:
                        best_weight = total_weight
                        best_node = i
            
            if best_node >= 0:
                partitions[p].append(best_node)
                assigned[best_node] = True
        
        # Grow partitions by adding nodes with highest connectivity to partition
        for _ in range(n - n_partitions):
            # Find smallest partition that isn't full
            smallest_p = min(range(n_partitions), 
                           key=lambda p: len(partitions[p]) if len(partitions[p]) < target_size + 1 else float('inf'))
            
            # Find best unassigned node for this partition
            best_node = -1
            best_gain = -float('inf')
            
            for i in range(n):
                if not assigned[i]:
                    # Compute gain = internal edges - external edges
                    internal = sum(graph.adjacency_matrix[i, j] for j in partitions[smallest_p])
                    gain = internal
                    if gain > best_gain:
                        best_gain = gain
                        best_node = i
            
            if best_node >= 0:
                partitions[smallest_p].append(best_node)
                assigned[best_node] = True
        
        return partitions
    
    @staticmethod
    def _uncoarsen_and_refine(original: LocalityGraph, coarse: LocalityGraph,
                              coarse_parts: List[List[int]], mapping: Dict[int, List[int]],
                              n_partitions: int, balance_factor: float) -> List[List[int]]:
        """
        Project partition to original graph and refine using FM algorithm.
        """
        # Project coarse partition to fine graph
        fine_parts = [[] for _ in range(n_partitions)]
        
        for p_idx, coarse_part in enumerate(coarse_parts):
            for coarse_node in coarse_part:
                if coarse_node in mapping:
                    fine_parts[p_idx].extend(mapping[coarse_node])
        
        # FM-style refinement
        fine_parts = METIS._fm_refine(original, fine_parts, balance_factor)
        
        return fine_parts
    
    @staticmethod
    def _fm_refine(graph: LocalityGraph, partitions: List[List[int]], 
                   balance_factor: float, max_passes: int = 10) -> List[List[int]]:
        """
        Fiduccia-Mattheyses refinement to reduce edge cut.
        """
        n = graph.num_tbs
        n_parts = len(partitions)
        target_size = n / n_parts
        max_size = int(target_size * balance_factor)
        min_size = int(target_size / balance_factor)
        
        # Build node to partition mapping
        node_part = {}
        for p_idx, part in enumerate(partitions):
            for node in part:
                node_part[node] = p_idx
        
        for _ in range(max_passes):
            improved = False
            
            # Try moving each node to best partition
            for node in range(n):
                current_part = node_part[node]
                
                if len(partitions[current_part]) <= min_size:
                    continue
                
                # Compute gain for moving to each other partition
                best_part = current_part
                best_gain = 0
                
                for target_part in range(n_parts):
                    if target_part == current_part:
                        continue
                    if len(partitions[target_part]) >= max_size:
                        continue
                    
                    # Gain = edges to target - edges to current
                    edges_to_current = sum(
                        graph.adjacency_matrix[node, j]
                        for j in partitions[current_part] if j != node
                    )
                    edges_to_target = sum(
                        graph.adjacency_matrix[node, j]
                        for j in partitions[target_part]
                    )
                    gain = edges_to_target - edges_to_current
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_part = target_part
                
                if best_part != current_part:
                    # Move node
                    partitions[current_part].remove(node)
                    partitions[best_part].append(node)
                    node_part[node] = best_part
                    improved = True
            
            if not improved:
                break
        
        return partitions


class KWay_TS:
    """
    K-Way Partition based Thread Block Scheduler (K-Way-TS).
    
    This scheduler partitions the TB graph into k parts (k = number of SMs),
    maximizing locality within each partition. TBs within each partition
    are then re-ordered using MST for optimal concurrent execution.
    
    Algorithm:
    1. Partition graph into k=num_SMs parts using METIS
    2. Each partition is assigned to one SM
    3. Re-order TBs within each partition using MST for L1 locality
    4. TBs execute in partition order, maximizing L1 hits
    
    Advantages over MST-TS:
    - Considers all edges, not just MST edges
    - Better L1 locality within partitions
    
    Disadvantages:
    - May lose L2 locality between partitions
    """
    
    def __init__(self, graph: LocalityGraph, gpu_config: GPUConfig):
        self.graph = graph
        self.config = gpu_config
    
    def partition(self) -> PartitionResult:
        """
        Perform K-Way partitioning.
        
        Returns:
            PartitionResult containing the TB schedule and assignments
        """
        n_sms = self.config.num_sms
        max_tb_per_sm = self.config.max_tb_per_sm
        n_tbs = self.graph.num_tbs
        
        # FIX: Calculate appropriate number of partitions
        # k should be the minimum of:
        #   - Number of SMs (original paper assumption for large kernels)
        #   - Number of partitions needed to hold all TBs (ceil(n_tbs / max_tb_per_sm))
        # This ensures we don't create more partitions than necessary for small kernels
        k = min(n_sms, max(1, (n_tbs + max_tb_per_sm - 1) // max_tb_per_sm))
        
        # Step 1: K-Way partition using METIS with corrected k
        metis_partitions = METIS.partition_graph(self.graph, k)
        
        # Step 1: K-Way partition using METIS
        # metis_partitions = METIS.partition_graph(self.graph, n_sms)
        
        # Step 2: Re-order TBs within each partition using MST
        partitions = []
        tb_schedule = []
        partition_assignment = {}
        sm_assignments = {sm: [] for sm in range(n_sms)}
        
        for p_idx, tb_list in enumerate(metis_partitions):
            if not tb_list:
                continue
            
            # Get subgraph for this partition
            subgraph = self.graph.get_subgraph(tb_list)
            
            # Order TBs within partition using MST
            local_ordering = PrimMST.get_mst_ordering(subgraph)
            ordered_tbs = [tb_list[i] for i in local_ordering]
            
            # Compute intra-partition locality
            locality = self._compute_partition_locality(ordered_tbs)
            
            # Create TB groups based on max_tb_per_sm
            x = self.config.max_tb_per_sm
            for i in range(0, len(ordered_tbs), x):
                group_tbs = ordered_tbs[i:i+x]
                group_locality = self._compute_partition_locality(group_tbs)
                group = TBGroup(group_tbs, p_idx, group_locality)
                partitions.append(group)
                sm_assignments[p_idx % n_sms].append(group)
            
            tb_schedule.extend(ordered_tbs)
            for tb in ordered_tbs:
                partition_assignment[tb] = p_idx
        
        # Compute locality metrics
        intra_locality = sum(self._compute_partition_locality(list(p)) 
                            for p in metis_partitions)
        total_locality = self.graph.get_total_edge_weight()
        inter_cut = total_locality - intra_locality
        
        return PartitionResult(
            method=PartitioningMethod.KWAY_TS,
            partitions=partitions,
            tb_schedule=tb_schedule,
            partition_assignment=partition_assignment,
            total_intra_partition_locality=intra_locality,
            total_inter_partition_cut=inter_cut,
            sm_assignments=sm_assignments
        )
    
    def _compute_partition_locality(self, tb_list: List[int]) -> float:
        """Compute sum of edge weights within a partition"""
        locality = 0.0
        for i, tb_i in enumerate(tb_list):
            for tb_j in tb_list[i+1:]:
                locality += self.graph.get_edge_weight(tb_i, tb_j)
        return locality


# ==============================================================================
# RB-TS: Recursive Bi-Partitioning based TB Scheduler
# ==============================================================================

class RB_TS:
    """
    Recursive Bi-Partitioning based Thread Block Scheduler (RB-TS).
    
    This scheduler recursively bi-partitions the graph until each partition
    is smaller than the maximum TBs per SM. This creates a binary tree where
    leaf nodes are TB groups that preserve both L1 and L2 locality.
    
    Algorithm (from paper):
    1. Let Q be the queue of TB groups
    2. Let L be the list of TB groups for SM scheduling
    3. Q.push(G₀)
    4. while Q not empty:
    5.     Gᵢ = Q.front(); Q.pop()
    6.     (G'ᵢ, G''ᵢ) = METIS.partition(Gᵢ, n=2)
    7.     if G'ᵢ.size() < maxTB: move G'ᵢ to L
    8.     else: Q.push(G'ᵢ)
    9.     if G''ᵢ.size() < maxTB: move G''ᵢ to L
    10.    else: Q.push(G''ᵢ)
    
    Advantages:
    - Preserves L1 locality (TBs grouped within same leaf)
    - Preserves L2 locality (adjacent groups share parent, scheduled together)
    - Better balanced than K-Way for varying SM workloads
    """
    
    def __init__(self, graph: LocalityGraph, gpu_config: GPUConfig):
        self.graph = graph
        self.config = gpu_config
    
    def partition(self) -> PartitionResult:
        """
        Perform Recursive Bi-Partitioning.
        
        Returns:
            PartitionResult containing the TB schedule and assignments
        """
        max_tb = self.config.max_tb_per_sm
        n_sms = self.config.num_sms
        
        # Queue for BFS-style recursive partitioning
        Q = deque()
        L = []  # Final leaf TB groups
        
        # Initialize with full graph
        initial_tbs = list(range(self.graph.num_tbs))
        Q.append(initial_tbs)
        
        # Recursive bi-partitioning
        while Q:
            current_group = Q.popleft()
            
            if len(current_group) <= max_tb:
                # This is a leaf node, add to L
                L.append(current_group)
            else:
                # Bi-partition this group
                subgraph = self.graph.get_subgraph(current_group)
                local_part0, local_part1 = METIS.bi_partition(subgraph)
                
                # Map back to original TB IDs
                part0 = [current_group[i] for i in local_part0]
                part1 = [current_group[i] for i in local_part1]
                
                # Check sizes and add to Q or L
                if len(part0) <= max_tb:
                    L.append(part0)
                else:
                    Q.append(part0)
                
                if len(part1) <= max_tb:
                    L.append(part1)
                else:
                    Q.append(part1)
        
        # Create TBGroups and assign to SMs
        partitions = []
        tb_schedule = []
        partition_assignment = {}
        sm_assignments = {sm: [] for sm in range(n_sms)}
        
        for p_idx, tb_list in enumerate(L):
            # Re-order within group using MST for max locality
            if len(tb_list) > 1:
                subgraph = self.graph.get_subgraph(tb_list)
                local_order = PrimMST.get_mst_ordering(subgraph)
                ordered_tbs = [tb_list[i] for i in local_order]
            else:
                ordered_tbs = tb_list
            
            locality = self._compute_group_locality(ordered_tbs)
            group = TBGroup(ordered_tbs, p_idx, locality)
            partitions.append(group)
            
            # Round-robin SM assignment for adjacent groups
            sm_idx = p_idx % n_sms
            sm_assignments[sm_idx].append(group)
            
            tb_schedule.extend(ordered_tbs)
            for tb in ordered_tbs:
                partition_assignment[tb] = p_idx
        
        # Compute locality metrics
        intra_locality = sum(p.total_locality for p in partitions)
        total_locality = self.graph.get_total_edge_weight()
        inter_cut = total_locality - intra_locality
        
        return PartitionResult(
            method=PartitioningMethod.RB_TS,
            partitions=partitions,
            tb_schedule=tb_schedule,
            partition_assignment=partition_assignment,
            total_intra_partition_locality=intra_locality,
            total_inter_partition_cut=inter_cut,
            sm_assignments=sm_assignments
        )
    
    def _compute_group_locality(self, tb_list: List[int]) -> float:
        """Compute sum of edge weights within a TB group"""
        locality = 0.0
        for i, tb_i in enumerate(tb_list):
            for tb_j in tb_list[i+1:]:
                locality += self.graph.get_edge_weight(tb_i, tb_j)
        return locality


# ==============================================================================
# Task Stealing for Load Balancing
# ==============================================================================

@dataclass
class SMState:
    """State of an SM during execution"""
    sm_id: int
    tb_queue: List[int]  # TBs waiting to execute
    next_idx: int = 0    # Points to next TB to execute
    tail_idx: int = 0    # Points to last TB in queue
    
    @property
    def waiting_tbs(self) -> int:
        """Number of TBs waiting in queue"""
        return self.tail_idx - self.next_idx
    
    def has_waiting_tbs(self) -> bool:
        return self.next_idx < self.tail_idx


class TaskStealer:
    """
    Task Stealing Algorithm for Load Balancing.
    
    When an SM finishes all its TBs early, it can steal TBs from a busy SM
    to balance the workload. This helps in the final phase of kernel execution.
    
    Algorithm (from paper):
    1. MaxWaitingTB = 0, AverageWaitingTB = 0
    2. For each SM:
    3.     WaitingTB = SM.tail - SM.next
    4.     if WaitingTB > MaxWaitingTB:
    5.         MaxWaitingTB = WaitingTB; DonorSM = SM
    6.     AverageWaitingTB += WaitingTB
    7. AverageWaitingTB /= total_SM
    8. StolenTBcount = MaxWaitingTB - AverageWaitingTB
    9. return DonorSM, StolenTBcount
    
    Note: Task stealing is used for K-Way-TS and RB-TS only.
    MST-TS implicitly load-balances through round-robin assignment.
    """
    
    def __init__(self, sm_states: Dict[int, SMState]):
        self.sm_states = sm_states
        self.num_sms = len(sm_states)
    
    def find_donor_and_steal_count(self) -> Tuple[Optional[int], int]:
        """
        Find the donor SM and number of TBs to steal.
        
        Returns:
            Tuple of (donor_sm_id, stolen_tb_count) or (None, 0) if no stealing needed
        """
        max_waiting = 0
        total_waiting = 0
        donor_sm = None
        
        for sm_id, state in self.sm_states.items():
            waiting = state.waiting_tbs
            total_waiting += waiting
            
            if waiting > max_waiting:
                max_waiting = waiting
                donor_sm = sm_id
        
        if self.num_sms == 0:
            return None, 0
        
        avg_waiting = total_waiting / self.num_sms
        stolen_count = int(max_waiting - avg_waiting)
        
        if stolen_count > 0 and donor_sm is not None:
            return donor_sm, stolen_count
        
        return None, 0
    
    def steal_tasks(self, recipient_sm: int) -> List[int]:
        """
        Perform task stealing for a recipient SM.
        
        Args:
            recipient_sm: SM ID that needs more work
            
        Returns:
            List of stolen TB IDs
        """
        donor_sm, steal_count = self.find_donor_and_steal_count()
        
        if donor_sm is None or steal_count <= 0:
            return []
        
        donor_state = self.sm_states[donor_sm]
        recipient_state = self.sm_states[recipient_sm]
        
        # Steal from tail of donor queue
        stolen_tbs = []
        for _ in range(steal_count):
            if donor_state.waiting_tbs <= 1:  # Keep at least 1 TB for donor
                break
            
            donor_state.tail_idx -= 1
            stolen_tb = donor_state.tb_queue[donor_state.tail_idx]
            stolen_tbs.append(stolen_tb)
        
        # Add to recipient queue
        recipient_state.tb_queue.extend(stolen_tbs)
        recipient_state.tail_idx += len(stolen_tbs)
        
        return stolen_tbs


# ==============================================================================
# Unified Partitioner Interface
# ==============================================================================

class PAVERPartitioner:
    """
    Unified interface for all PAVER graph partitioning methods.
    
    This class provides a simple interface to select and run any of the
    three partitioning algorithms (MST-TS, K-Way-TS, RB-TS) and optionally
    enable task stealing for load balancing.
    
    Usage:
        graph = LocalityGraph(...)  # From LocalityGuru
        config = GPUConfig(num_sms=15, max_tb_per_sm=8)
        
        partitioner = PAVERPartitioner(graph, config)
        result = partitioner.partition(PartitioningMethod.RB_TS)
        
        # Get TB schedule for GPU runtime
        schedule = result.tb_schedule
    """
    
    def __init__(self, graph: LocalityGraph, gpu_config: GPUConfig):
        """
        Initialize the PAVER partitioner.
        
        Args:
            graph: LocalityGraph from LocalityGuru analysis
            gpu_config: GPU hardware configuration
        """
        self.graph = graph
        self.config = gpu_config
    
    def partition(self, method: PartitioningMethod = PartitioningMethod.RB_TS) -> PartitionResult:
        """
        Partition the locality graph using the specified method.
        
        Args:
            method: Partitioning method to use (MST_TS, KWAY_TS, or RB_TS)
            
        Returns:
            PartitionResult with TB schedule and assignments
        """
        if method == PartitioningMethod.MST_TS:
            partitioner = MST_TS(self.graph, self.config)
        elif method == PartitioningMethod.KWAY_TS:
            partitioner = KWay_TS(self.graph, self.config)
        elif method == PartitioningMethod.RB_TS:
            partitioner = RB_TS(self.graph, self.config)
        else:
            raise ValueError(f"Unknown partitioning method: {method}")
        
        return partitioner.partition()
    
    def partition_all(self) -> Dict[PartitioningMethod, PartitionResult]:
        """
        Run all partitioning methods and return results for comparison.
        
        Returns:
            Dict mapping method to its PartitionResult
        """
        results = {}
        for method in PartitioningMethod:
            results[method] = self.partition(method)
        return results
    
    def compare_methods(self) -> str:
        """
        Compare all partitioning methods and return a summary.
        
        Returns:
            Formatted string comparing the methods
        """
        results = self.partition_all()
        
        lines = [
            "=" * 70,
            "PAVER Graph Partitioning Comparison",
            "=" * 70,
            f"Graph: {self.graph.num_tbs} TBs, {self.graph.get_total_edge_weight():.0f} total edge weight",
            f"GPU Config: {self.config.num_sms} SMs, max {self.config.max_tb_per_sm} TBs/SM",
            "-" * 70,
            f"{'Method':<15} {'Partitions':<12} {'Intra-Locality':<18} {'Edge Cut':<15}",
            "-" * 70,
        ]
        
        for method, result in results.items():
            lines.append(
                f"{method.value:<15} {len(result.partitions):<12} "
                f"{result.total_intra_partition_locality:<18.2f} "
                f"{result.total_inter_partition_cut:<15.2f}"
            )
        
        lines.append("=" * 70)
        
        # Recommendation
        best_method = max(results.keys(), 
                        key=lambda m: results[m].total_intra_partition_locality)
        lines.append(f"Recommended: {best_method.value} (highest intra-partition locality)")
        
        return "\n".join(lines)


# ==============================================================================
# Utility Functions for Integration with LocalityGuru
# ==============================================================================

def load_locality_matrix(filepath: str) -> np.ndarray:
    """
    Load locality matrix from file (CSV or NPY format).
    
    Args:
        filepath: Path to locality matrix file
        
    Returns:
        numpy array of the adjacency matrix
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.csv'):
        return np.loadtxt(filepath, delimiter=',')
    else:
        # Try to load as text
        return np.loadtxt(filepath)


def create_graph_from_locality_guru(locality_matrix: np.ndarray, 
                                     tb_ids: Optional[List[int]] = None) -> LocalityGraph:
    """
    Create a LocalityGraph from LocalityGuru output.
    
    Args:
        locality_matrix: NxN adjacency matrix from LocalityGuru
        tb_ids: Optional list of TB IDs (default: 0 to N-1)
        
    Returns:
        LocalityGraph ready for partitioning
    """
    n = locality_matrix.shape[0]
    if tb_ids is None:
        tb_ids = list(range(n))
    
    return LocalityGraph(n, locality_matrix.copy(), tb_ids)


def export_schedule_for_gpu(result: PartitionResult, output_file: str):
    """
    Export TB schedule in format readable by GPU runtime.
    
    The output format is compatible with PAVER's hardware implementation
    which stores TB groups in global memory.
    
    Args:
        result: PartitionResult from partitioning
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# PAVER TB Schedule - Method: {result.method.value}\n")
        f.write(f"# Total Partitions: {len(result.partitions)}\n")
        f.write(f"# Intra-partition Locality: {result.total_intra_partition_locality:.2f}\n")
        f.write(f"# Inter-partition Cut: {result.total_inter_partition_cut:.2f}\n")
        f.write("#\n")
        
        # Write partition assignments
        f.write("# Format: PARTITION_ID: TB_ID, TB_ID, ...\n")
        for group in result.partitions:
            tb_str = ",".join(map(str, group.tb_list))
            f.write(f"{group.partition_id}: {tb_str}\n")
        
        # Write SM assignments
        f.write("#\n# SM Assignments:\n")
        for sm_id, groups in result.sm_assignments.items():
            group_ids = [g.partition_id for g in groups]
            f.write(f"# SM {sm_id}: Partitions {group_ids}\n")


# ==============================================================================
# Main / Testing
# ==============================================================================

if __name__ == "__main__":
    # Example usage with synthetic locality graph
    print("PAVER Graph Partitioning Module")
    print("=" * 50)
    
    # Create a sample locality graph (e.g., from matrix multiplication)
    # This simulates row-wise sharing pattern
    n_tbs = 64
    adj_matrix = np.zeros((n_tbs, n_tbs))
    
    # Create row-wise locality pattern (like MM)
    grid_dim = int(np.sqrt(n_tbs))
    for i in range(n_tbs):
        row_i = i // grid_dim
        for j in range(n_tbs):
            row_j = j // grid_dim
            if row_i == row_j and i != j:
                # Same row shares data
                adj_matrix[i, j] = 100 + np.random.randint(0, 50)
            elif abs(row_i - row_j) == 1 and i != j:
                # Adjacent rows share some data
                adj_matrix[i, j] = np.random.randint(10, 30)
    
    # Make symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # Create graph and config
    graph = LocalityGraph(n_tbs, adj_matrix)
    config = GPUConfig(num_sms=15, max_tb_per_sm=8)
    
    print(f"\nCreated locality graph:")
    print(f"  - {n_tbs} Thread Blocks")
    print(f"  - Total edge weight: {graph.get_total_edge_weight():.0f}")
    print(f"  - GPU: {config.num_sms} SMs, max {config.max_tb_per_sm} TBs/SM")
    
    # Run partitioner
    partitioner = PAVERPartitioner(graph, config)
    
    print("\n" + partitioner.compare_methods())
    
    # Detailed results for best method
    print("\nDetailed RB-TS Result:")
    result = partitioner.partition(PartitioningMethod.RB_TS)
    print(f"  - Number of partitions: {len(result.partitions)}")
    print(f"  - First 5 partitions:")
    for i, group in enumerate(result.partitions[:5]):
        print(f"    Partition {i}: TBs {group.tb_list[:5]}{'...' if len(group.tb_list) > 5 else ''}")
        print(f"               Locality: {group.total_locality:.2f}")
