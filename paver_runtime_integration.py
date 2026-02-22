#!/usr/bin/env python3
"""
PAVER Runtime Integration Module

This module integrates LocalityGuru analysis with graph partitioning algorithms
to provide a complete TB scheduling solution. It handles:

1. Loading/converting LocalityGuru output to LocalityGraph
2. Applying graph partitioning (MST-TS, K-Way-TS, RB-TS)  
3. Generating scheduler configuration for GPGPU-Sim or hardware
4. Runtime task stealing for load balancing

Reference:
    D. Tripathy et al., "PAVER: Locality Graph-Based Thread Block Scheduling for GPUs"
    ACM Transactions on Architecture and Code Optimization, Vol. 18, No. 3, 2021
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os

# Import from graph partitioning module
from paver_graph_partitioning import (
    LocalityGraph, GPUConfig, TBGroup, PartitionResult,
    PartitioningMethod, PAVERPartitioner, TaskStealer, SMState,
    PrimMST, MST_TS, KWay_TS, RB_TS
)


# ==============================================================================
# GPU Architecture Configurations
# ==============================================================================

class GPUArchitecture(Enum):
    """Supported GPU architectures"""
    FERMI = "fermi"       # GTX 480
    PASCAL = "pascal"     # TITAN X
    VOLTA = "volta"       # TITAN V
    AMPERE = "ampere"     # RTX 3090
    CUSTOM = "custom"


# Default configurations from PAVER paper (Table 1)
GPU_CONFIGS = {
    GPUArchitecture.FERMI: {
        "num_sms": 15,
        "max_threads_per_sm": 1536,
        "max_tb_per_sm": 8,
        "l1_cache_size_kb": 16,  # Configurable 16/48 KB
        "l2_cache_size_kb": 768,
        "shared_mem_per_sm_kb": 48,
        "registers_per_sm": 32768,
        "warp_size": 32,
    },
    GPUArchitecture.PASCAL: {
        "num_sms": 28,
        "max_threads_per_sm": 2048,
        "max_tb_per_sm": 32,
        "l1_cache_size_kb": 48,
        "l2_cache_size_kb": 3072,
        "shared_mem_per_sm_kb": 96,
        "registers_per_sm": 65536,
        "warp_size": 32,
    },
    GPUArchitecture.VOLTA: {
        "num_sms": 80,
        "max_threads_per_sm": 2048,
        "max_tb_per_sm": 32,
        "l1_cache_size_kb": 128,  # Combined L1/shared
        "l2_cache_size_kb": 6144,
        "shared_mem_per_sm_kb": 96,
        "registers_per_sm": 65536,
        "warp_size": 32,
    },
    GPUArchitecture.AMPERE: {
        "num_sms": 82,
        "max_threads_per_sm": 2048,
        "max_tb_per_sm": 32,
        "l1_cache_size_kb": 192,
        "l2_cache_size_kb": 6144,
        "shared_mem_per_sm_kb": 164,
        "registers_per_sm": 65536,
        "warp_size": 32,
    },
}


def get_gpu_config(arch: GPUArchitecture, **overrides) -> GPUConfig:
    """
    Get GPU configuration for specified architecture.
    
    Args:
        arch: GPU architecture
        **overrides: Override specific config values
        
    Returns:
        GPUConfig instance
    """
    if arch == GPUArchitecture.CUSTOM:
        # Must provide all values
        return GPUConfig(**overrides)
    
    config = GPU_CONFIGS[arch].copy()
    config.update(overrides)
    
    return GPUConfig(
        num_sms=config["num_sms"],
        max_tb_per_sm=config.get("max_tb_per_sm", 8),
        l1_cache_size_kb=config.get("l1_cache_size_kb", 48),
        l2_cache_size_kb=config.get("l2_cache_size_kb", 1536),
    )


# ==============================================================================
# Kernel Resource Analysis
# ==============================================================================

@dataclass
class KernelResources:
    """
    Resource requirements for a CUDA kernel.
    
    Used to compute the actual max_tb_per_sm based on resource constraints.
    """
    threads_per_tb: int          # Total threads per TB
    registers_per_thread: int    # Registers used per thread  
    shared_mem_per_tb_bytes: int # Shared memory per TB
    
    def compute_max_tb_per_sm(self, arch: GPUArchitecture) -> int:
        """
        Compute maximum concurrent TBs per SM based on resources.
        
        The limiting factor is whichever resource runs out first:
        - Thread slots
        - Registers  
        - Shared memory
        - Hardware TB limit
        
        Args:
            arch: GPU architecture
            
        Returns:
            Maximum TBs per SM for this kernel
        """
        config = GPU_CONFIGS.get(arch, GPU_CONFIGS[GPUArchitecture.FERMI])
        
        # Hardware limit
        max_hw = config["max_tb_per_sm"]
        
        # Thread limit
        max_threads = config["max_threads_per_sm"] // self.threads_per_tb
        
        # Register limit
        regs_per_tb = self.threads_per_tb * self.registers_per_thread
        max_regs = config["registers_per_sm"] // regs_per_tb if regs_per_tb > 0 else max_hw
        
        # Shared memory limit
        shared_mem_total = config["shared_mem_per_sm_kb"] * 1024
        max_smem = shared_mem_total // self.shared_mem_per_tb_bytes if self.shared_mem_per_tb_bytes > 0 else max_hw
        
        return min(max_hw, max_threads, max_regs, max_smem)


# ==============================================================================
# LocalityGuru Integration
# ==============================================================================

@dataclass
class LocalityGuruResult:
    """
    Result from LocalityGuru PTX analysis.
    
    Contains all information needed to construct the locality graph.
    """
    kernel_name: str
    grid_dim: Tuple[int, int, int]  # (x, y, z)
    block_dim: Tuple[int, int, int]  # (x, y, z)
    num_tbs: int
    locality_matrix: np.ndarray
    tb_address_map: Optional[Dict[int, List[int]]] = None  # TB -> memory addresses
    
    @property
    def total_threads(self) -> int:
        return self.num_tbs * np.prod(self.block_dim)


def load_locality_guru_output(filepath: str) -> LocalityGuruResult:
    """
    Load LocalityGuru analysis output from file.
    
    Supports multiple formats:
    - .npy: Just the locality matrix
    - .npz: Matrix + metadata
    - .json: Full result with all metadata
    
    Args:
        filepath: Path to LocalityGuru output file
        
    Returns:
        LocalityGuruResult with parsed data
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npy':
        matrix = np.load(filepath)
        n_tbs = matrix.shape[0]
        # Estimate grid dimensions
        grid_side = int(np.sqrt(n_tbs))
        if grid_side * grid_side == n_tbs:
            grid_dim = (grid_side, grid_side, 1)
        else:
            grid_dim = (n_tbs, 1, 1)
        
        return LocalityGuruResult(
            kernel_name="unknown",
            grid_dim=grid_dim,
            block_dim=(32, 1, 1),  # Default
            num_tbs=n_tbs,
            locality_matrix=matrix
        )
    
    elif ext == '.npz':
        data = np.load(filepath, allow_pickle=True)
        matrix = data['locality_matrix']
        metadata = data.get('metadata', {}).item() if 'metadata' in data else {}
        
        return LocalityGuruResult(
            kernel_name=metadata.get('kernel_name', 'unknown'),
            grid_dim=tuple(metadata.get('grid_dim', (matrix.shape[0], 1, 1))),
            block_dim=tuple(metadata.get('block_dim', (32, 1, 1))),
            num_tbs=matrix.shape[0],
            locality_matrix=matrix,
            tb_address_map=metadata.get('tb_address_map', None)
        )
    
    elif ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both raw matrix arrays and dictionary format
        if isinstance(data, list):
            # Raw matrix array (direct LocalityGuru output)
            matrix = np.array(data)
            
            # Try to extract kernel name and grid dims from filename
            # Expected format: kernelname_gridX-gridY_matrix.json
            filename = os.path.basename(filepath)
            kernel_name = 'unknown'
            grid_dim = None
            
            # Parse filename like "_Z11gemm_kerneliiiffPfS_S__2-8_matrix.json"
            if '_matrix.json' in filename:
                parts = filename.replace('_matrix.json', '')
                # Find grid dimensions (e.g., "2-8" at the end)
                if '-' in parts:
                    segments = parts.rsplit('_', 1)
                    if len(segments) == 2:
                        kernel_name = segments[0]
                        grid_str = segments[1]
                        try:
                            grid_parts = grid_str.split('-')
                            if len(grid_parts) == 2:
                                gx, gy = int(grid_parts[0]), int(grid_parts[1])
                                grid_dim = (gx, gy, 1)
                        except ValueError:
                            pass
            
            block_dim = (32, 1, 1)
            tb_address_map = None
        else:
            # Dictionary format with metadata
            matrix = np.array(data['locality_matrix'])
            kernel_name = data.get('kernel_name', 'unknown')
            grid_dim = tuple(data.get('grid_dim', None)) if data.get('grid_dim') else None
            block_dim = tuple(data.get('block_dim', [32, 1, 1]))
            tb_address_map = data.get('tb_address_map', None)
        
        # Infer grid dimensions if not provided
        n_tbs = matrix.shape[0]
        if grid_dim is None:
            grid_side = int(np.sqrt(n_tbs))
            grid_dim = (grid_side, grid_side, 1) if grid_side * grid_side == n_tbs else (n_tbs, 1, 1)
        
        return LocalityGuruResult(
            kernel_name=kernel_name,
            grid_dim=grid_dim,
            block_dim=block_dim,
            num_tbs=n_tbs,
            locality_matrix=matrix,
            tb_address_map=tb_address_map
        )
    
    else:
        # Try loading as text (CSV-like)
        matrix = np.loadtxt(filepath, delimiter=',')
        n_tbs = matrix.shape[0]
        
        return LocalityGuruResult(
            kernel_name="unknown",
            grid_dim=(n_tbs, 1, 1),
            block_dim=(32, 1, 1),
            num_tbs=n_tbs,
            locality_matrix=matrix
        )


def create_locality_graph_from_guru(guru_result: LocalityGuruResult) -> LocalityGraph:
    """
    Convert LocalityGuru result to LocalityGraph for partitioning.
    
    Args:
        guru_result: Output from LocalityGuru analysis
        
    Returns:
        LocalityGraph ready for PAVER partitioning
    """
    # Ensure matrix is symmetric (undirected graph)
    matrix = guru_result.locality_matrix.copy()
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2
    
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(matrix, 0)
    
    return LocalityGraph(
        num_tbs=guru_result.num_tbs,
        adjacency_matrix=matrix,
        tb_ids=list(range(guru_result.num_tbs))
    )


# ==============================================================================
# PAVER Scheduler Configuration Generation
# ==============================================================================

@dataclass
class SchedulerConfig:
    """
    Configuration for PAVER TB scheduler.
    
    This can be exported for use by GPGPU-Sim or hardware implementation.
    """
    method: PartitioningMethod
    num_partitions: int
    partition_list: List[List[int]]  # List of TB lists per partition
    sm_assignments: Dict[int, List[int]]  # SM ID -> partition IDs
    tb_schedule: List[int]  # Flattened TB execution order
    enable_task_stealing: bool = True
    
    def to_gpgpu_sim_format(self) -> str:
        """
        Export configuration for GPGPU-Sim.
        
        Format is compatible with PAVER's GPGPU-Sim modifications.
        """
        lines = []
        lines.append(f"# PAVER Scheduler Config")
        lines.append(f"# Method: {self.method.value}")
        lines.append(f"paver_enabled 1")
        lines.append(f"paver_method {self.method.value}")
        lines.append(f"paver_task_stealing {1 if self.enable_task_stealing else 0}")
        lines.append(f"paver_num_partitions {self.num_partitions}")
        lines.append("")
        
        # TB schedule (flattened)
        lines.append(f"# TB Schedule ({len(self.tb_schedule)} TBs)")
        lines.append(f"paver_tb_schedule " + ",".join(map(str, self.tb_schedule)))
        lines.append("")
        
        # Partition definitions
        lines.append("# Partition definitions")
        for i, partition in enumerate(self.partition_list):
            lines.append(f"paver_partition {i} " + ",".join(map(str, partition)))
        
        lines.append("")
        
        # SM assignments
        lines.append("# SM to partition mapping")
        for sm_id, part_ids in self.sm_assignments.items():
            lines.append(f"paver_sm_assign {sm_id} " + ",".join(map(str, part_ids)))
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization"""
        return {
            "method": self.method.value,
            "num_partitions": self.num_partitions,
            "partition_list": self.partition_list,
            "sm_assignments": {str(k): v for k, v in self.sm_assignments.items()},
            "tb_schedule": self.tb_schedule,
            "enable_task_stealing": self.enable_task_stealing
        }


def create_scheduler_config(result: PartitionResult, 
                           enable_task_stealing: bool = True) -> SchedulerConfig:
    """
    Create scheduler configuration from partition result.
    
    Args:
        result: PartitionResult from graph partitioning
        enable_task_stealing: Whether to enable task stealing
        
    Returns:
        SchedulerConfig ready for export
    """
    partition_list = [list(p.tb_list) for p in result.partitions]
    sm_assignments = {
        sm_id: [g.partition_id for g in groups]
        for sm_id, groups in result.sm_assignments.items()
    }
    
    return SchedulerConfig(
        method=result.method,
        num_partitions=len(result.partitions),
        partition_list=partition_list,
        sm_assignments=sm_assignments,
        tb_schedule=result.tb_schedule,
        enable_task_stealing=enable_task_stealing
    )


# ==============================================================================
# Runtime Simulator for Validation
# ==============================================================================

class PAVERRuntimeSimulator:
    """
    Simulates PAVER TB scheduling at runtime.
    
    This is useful for validating partitioning results and
    estimating performance benefits before actual execution.
    """
    
    def __init__(self, config: SchedulerConfig, gpu_config: GPUConfig):
        self.sched_config = config
        self.gpu_config = gpu_config
        
        # Initialize SM states
        self.sm_states = {}
        for sm_id in range(gpu_config.num_sms):
            if sm_id in config.sm_assignments:
                # Get TBs for this SM from assigned partitions
                tbs = []
                for part_id in config.sm_assignments[sm_id]:
                    if part_id < len(config.partition_list):
                        tbs.extend(config.partition_list[part_id])
                
                self.sm_states[sm_id] = SMState(
                    sm_id=sm_id,
                    tb_queue=tbs,
                    next_idx=0,
                    tail_idx=len(tbs)
                )
            else:
                self.sm_states[sm_id] = SMState(
                    sm_id=sm_id,
                    tb_queue=[],
                    next_idx=0,
                    tail_idx=0
                )
        
        self.task_stealer = TaskStealer(self.sm_states)
        self.executed_tbs = []
        self.steal_events = []
    
    def simulate_execution(self, tb_execution_times: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
        """
        Simulate kernel execution with PAVER scheduling.
        
        Args:
            tb_execution_times: Optional dict of TB ID -> execution cycles
                               If None, assumes uniform execution time
                               
        Returns:
            Dict with simulation statistics
        """
        if tb_execution_times is None:
            # Assume uniform execution
            total_tbs = sum(s.tail_idx for s in self.sm_states.values())
            tb_execution_times = {i: 1000 for i in range(total_tbs)}
        
        # Track execution
        sm_current_tb = {sm_id: None for sm_id in self.sm_states.keys()}
        sm_finish_time = {sm_id: 0 for sm_id in self.sm_states.keys()}
        
        current_time = 0
        max_concurrent = self.gpu_config.max_tb_per_sm
        
        while True:
            # Check for completed TBs and issue new ones
            all_done = True
            
            for sm_id, state in self.sm_states.items():
                # Count active TBs on this SM
                active_tbs = 1 if sm_current_tb[sm_id] is not None else 0
                
                # Issue new TBs if capacity available
                while active_tbs < max_concurrent and state.next_idx < state.tail_idx:
                    tb_id = state.tb_queue[state.next_idx]
                    state.next_idx += 1
                    self.executed_tbs.append((tb_id, sm_id, current_time))
                    active_tbs += 1
                    all_done = False
                
                if state.waiting_tbs > 0:
                    all_done = False
            
            if all_done:
                break
            
            # Task stealing check
            for sm_id, state in self.sm_states.items():
                if state.waiting_tbs == 0 and self.sched_config.enable_task_stealing:
                    stolen = self.task_stealer.steal_tasks(sm_id)
                    if stolen:
                        self.steal_events.append((current_time, sm_id, stolen))
            
            current_time += 1
            
            # Prevent infinite loop
            if current_time > 1000000:
                break
        
        return {
            "total_cycles": current_time,
            "tbs_executed": len(self.executed_tbs),
            "steal_events": len(self.steal_events),
            "execution_trace": self.executed_tbs[:100],  # First 100 for debugging
        }


# ==============================================================================
# Complete PAVER Pipeline
# ==============================================================================

class PAVERPipeline:
    """
    Complete PAVER pipeline from LocalityGuru to scheduler config.
    
    This class provides a simple interface for the full PAVER workflow:
    1. Load LocalityGuru output
    2. Configure GPU parameters
    3. Run graph partitioning
    4. Generate scheduler configuration
    5. Optionally simulate execution
    
    Usage:
        pipeline = PAVERPipeline()
        pipeline.load_locality_data("locality_matrix.npy")
        pipeline.configure_gpu(GPUArchitecture.VOLTA, max_tb_per_sm=6)
        result = pipeline.run_partitioning(PartitioningMethod.RB_TS)
        pipeline.export_scheduler_config("paver_config.txt")
    """
    
    def __init__(self):
        self.guru_result: Optional[LocalityGuruResult] = None
        self.locality_graph: Optional[LocalityGraph] = None
        self.gpu_config: Optional[GPUConfig] = None
        self.partition_result: Optional[PartitionResult] = None
        self.scheduler_config: Optional[SchedulerConfig] = None
    
    def load_locality_data(self, filepath: str) -> 'PAVERPipeline':
        """
        Load LocalityGuru analysis output.
        
        Args:
            filepath: Path to locality data file
            
        Returns:
            self for method chaining
        """
        self.guru_result = load_locality_guru_output(filepath)
        self.locality_graph = create_locality_graph_from_guru(self.guru_result)
        print(f"Loaded locality graph: {self.guru_result.num_tbs} TBs, "
              f"kernel: {self.guru_result.kernel_name}")
        return self
    
    def load_locality_matrix(self, matrix: np.ndarray, 
                            kernel_name: str = "unknown",
                            grid_dim: Tuple[int, int, int] = None,
                            block_dim: Tuple[int, int, int] = (32, 1, 1)) -> 'PAVERPipeline':
        """
        Load locality matrix directly (e.g., from in-memory LocalityGuru result).
        
        Args:
            matrix: NxN locality adjacency matrix
            kernel_name: Name of the kernel
            grid_dim: Grid dimensions (x, y, z)
            block_dim: Block dimensions (x, y, z)
            
        Returns:
            self for method chaining
        """
        n_tbs = matrix.shape[0]
        if grid_dim is None:
            grid_side = int(np.sqrt(n_tbs))
            grid_dim = (grid_side, grid_side, 1) if grid_side*grid_side == n_tbs else (n_tbs, 1, 1)
        
        self.guru_result = LocalityGuruResult(
            kernel_name=kernel_name,
            grid_dim=grid_dim,
            block_dim=block_dim,
            num_tbs=n_tbs,
            locality_matrix=matrix
        )
        self.locality_graph = create_locality_graph_from_guru(self.guru_result)
        print(f"Loaded locality graph: {n_tbs} TBs, kernel: {kernel_name}")
        return self
    
    def configure_gpu(self, arch: GPUArchitecture = GPUArchitecture.VOLTA,
                     kernel_resources: Optional[KernelResources] = None,
                     **overrides) -> 'PAVERPipeline':
        """
        Configure GPU parameters.
        
        Args:
            arch: GPU architecture
            kernel_resources: Kernel resource requirements (to compute actual max_tb_per_sm)
            **overrides: Override specific config values
            
        Returns:
            self for method chaining
        """
        self.gpu_config = get_gpu_config(arch, **overrides)
        
        # Adjust max_tb_per_sm based on kernel resources if provided
        if kernel_resources is not None:
            actual_max = kernel_resources.compute_max_tb_per_sm(arch)
            self.gpu_config.max_tb_per_sm = min(self.gpu_config.max_tb_per_sm, actual_max)
            print(f"Adjusted max_tb_per_sm to {self.gpu_config.max_tb_per_sm} based on kernel resources")
        
        print(f"GPU config: {arch.value}, {self.gpu_config.num_sms} SMs, "
              f"max {self.gpu_config.max_tb_per_sm} TBs/SM")
        return self
    
    def run_partitioning(self, method: PartitioningMethod = PartitioningMethod.RB_TS) -> PartitionResult:
        """
        Run graph partitioning with specified method.
        
        Args:
            method: Partitioning method (MST_TS, KWAY_TS, RB_TS)
            
        Returns:
            PartitionResult with schedule and assignments
        """
        if self.locality_graph is None:
            raise ValueError("Must load locality data first")
        if self.gpu_config is None:
            print("Warning: GPU not configured, using default Volta config")
            self.configure_gpu(GPUArchitecture.VOLTA)
        
        partitioner = PAVERPartitioner(self.locality_graph, self.gpu_config)
        self.partition_result = partitioner.partition(method)
        
        print(f"\nPartitioning complete ({method.value}):")
        print(f"  - Partitions: {len(self.partition_result.partitions)}")
        print(f"  - Intra-partition locality: {self.partition_result.total_intra_partition_locality:.2f}")
        print(f"  - Inter-partition cut: {self.partition_result.total_inter_partition_cut:.2f}")
        
        return self.partition_result
    
    def compare_all_methods(self) -> Dict[PartitioningMethod, PartitionResult]:
        """
        Run all partitioning methods and compare results.
        
        Returns:
            Dict mapping method to result
        """
        if self.locality_graph is None:
            raise ValueError("Must load locality data first")
        if self.gpu_config is None:
            self.configure_gpu(GPUArchitecture.VOLTA)
        
        partitioner = PAVERPartitioner(self.locality_graph, self.gpu_config)
        print("\n" + partitioner.compare_methods())
        return partitioner.partition_all()
    
    def create_scheduler_config(self, enable_task_stealing: bool = True) -> SchedulerConfig:
        """
        Create scheduler configuration from partitioning result.
        
        Args:
            enable_task_stealing: Whether to enable task stealing
            
        Returns:
            SchedulerConfig ready for export
        """
        if self.partition_result is None:
            raise ValueError("Must run partitioning first")
        
        self.scheduler_config = create_scheduler_config(
            self.partition_result, 
            enable_task_stealing
        )
        return self.scheduler_config
    
    def export_scheduler_config(self, filepath: str, format: str = 'gpgpu_sim') -> 'PAVERPipeline':
        """
        Export scheduler configuration to file.
        
        Args:
            filepath: Output file path
            format: 'gpgpu_sim' or 'json'
            
        Returns:
            self for method chaining
        """
        if self.scheduler_config is None:
            self.create_scheduler_config()
        
        if format == 'gpgpu_sim':
            content = self.scheduler_config.to_gpgpu_sim_format()
        elif format == 'json':
            content = json.dumps(self.scheduler_config.to_dict(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Exported scheduler config to {filepath}")
        return self
    
    def simulate_execution(self) -> Dict[str, Any]:
        """
        Run simulation of PAVER scheduling.
        
        Returns:
            Simulation statistics
        """
        if self.scheduler_config is None:
            self.create_scheduler_config()
        
        simulator = PAVERRuntimeSimulator(self.scheduler_config, self.gpu_config)
        stats = simulator.simulate_execution()
        
        print(f"\nSimulation results:")
        print(f"  - Total cycles: {stats['total_cycles']}")
        print(f"  - TBs executed: {stats['tbs_executed']}")
        print(f"  - Task steal events: {stats['steal_events']}")
        
        return stats
    
    def get_summary(self) -> str:
        """
        Get summary of current pipeline state.
        
        Returns:
            Formatted summary string
        """
        lines = ["=" * 60, "PAVER Pipeline Summary", "=" * 60]
        
        if self.guru_result:
            lines.append(f"Kernel: {self.guru_result.kernel_name}")
            lines.append(f"Grid: {self.guru_result.grid_dim}")
            lines.append(f"Block: {self.guru_result.block_dim}")
            lines.append(f"Total TBs: {self.guru_result.num_tbs}")
        
        if self.locality_graph:
            lines.append(f"Total edge weight: {self.locality_graph.get_total_edge_weight():.2f}")
        
        if self.gpu_config:
            lines.append(f"\nGPU: {self.gpu_config.num_sms} SMs, "
                        f"max {self.gpu_config.max_tb_per_sm} TBs/SM")
        
        if self.partition_result:
            lines.append(f"\nPartitioning ({self.partition_result.method.value}):")
            lines.append(f"  Partitions: {len(self.partition_result.partitions)}")
            lines.append(f"  Intra-locality: {self.partition_result.total_intra_partition_locality:.2f}")
            lines.append(f"  Edge cut: {self.partition_result.total_inter_partition_cut:.2f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ==============================================================================
# Main / Testing
# ==============================================================================

if __name__ == "__main__":
    print("PAVER Runtime Integration Module")
    print("=" * 60)
    
    # Create sample locality matrix (matrix multiplication pattern)
    n_tbs = 64
    grid_dim = (8, 8, 1)
    
    # Generate MM-like locality pattern
    matrix = np.zeros((n_tbs, n_tbs))
    grid_x, grid_y = 8, 8
    
    for i in range(n_tbs):
        row_i = i // grid_x
        col_i = i % grid_x
        for j in range(n_tbs):
            row_j = j // grid_x
            col_j = j % grid_x
            
            if i == j:
                continue
            
            # Same row shares A matrix data
            if row_i == row_j:
                matrix[i, j] += 256  # N shared elements
            
            # Same column shares B matrix data
            if col_i == col_j:
                matrix[i, j] += 256
    
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    
    # Run pipeline
    pipeline = PAVERPipeline()
    
    # Load data
    pipeline.load_locality_matrix(
        matrix,
        kernel_name="MatrixMultiplication",
        grid_dim=grid_dim,
        block_dim=(32, 8, 1)
    )
    
    # Configure GPU (Fermi for comparison with paper)
    pipeline.configure_gpu(
        GPUArchitecture.FERMI,
        kernel_resources=KernelResources(
            threads_per_tb=256,
            registers_per_thread=32,
            shared_mem_per_tb_bytes=0
        )
    )
    
    # Compare methods
    pipeline.compare_all_methods()
    
    # Run best method (RB-TS)
    result = pipeline.run_partitioning(PartitioningMethod.RB_TS)
    
    # Create and export config
    pipeline.create_scheduler_config(enable_task_stealing=False)
    pipeline.export_scheduler_config("/home/claude/paver_scheduler_config.txt", format='gpgpu_sim')
    pipeline.export_scheduler_config("/home/claude/paver_scheduler_config.json", format='json')
    
    # Simulate
    pipeline.simulate_execution()
    
    # Print summary
    print("\n" + pipeline.get_summary())
