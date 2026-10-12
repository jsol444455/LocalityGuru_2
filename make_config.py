"""Turn one LocalityGuru locality matrix into a GPGPU-Sim PAVER config.

Usage (from the LocalityGuru_2 directory):
  python3 make_config.py <matrix.json> <METHOD> <output.txt> [interleaved]

METHOD: KWAY_TS, MST_TS, RB_TS, TSP_TS or RB_TSP_TS
GPU:    Titan V config of PAVER_GPGPU-SIM: 80 SMs, at most 8 TBs (256 threads) per SM
"""
import sys
from paver_runtime_integration import PAVERPipeline, GPUArchitecture, PartitioningMethod

matrix_file, method, out = sys.argv[1], sys.argv[2], sys.argv[3]
sm_order = "interleaved" if len(sys.argv) > 4 and sys.argv[4] == "interleaved" else None

p = PAVERPipeline()
p.load_locality_data(matrix_file)
p.configure_gpu(GPUArchitecture.VOLTA, max_tb_per_sm=8, num_sms=80)
p.run_partitioning(getattr(PartitioningMethod, method))
p.create_scheduler_config(enable_task_stealing=True, sm_order=sm_order)
p.export_scheduler_config(out)
