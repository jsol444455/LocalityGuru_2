from paver_runtime_integration import PAVERPipeline, GPUArchitecture, PartitioningMethod

# Initialize pipeline
pipeline = PAVERPipeline()

# Load your LocalityGuru output
pipeline.load_locality_data("/home/m/LocalityGuru_2/img/gramschmidt/_Z19gramschmidt_kernel1PfS_S_i_8-1_matrix.json")
# OR load matrix directly
# pipeline.load_locality_matrix(locality_matrix, kernel_name="GEMM")

# Configure GPU
pipeline.configure_gpu(GPUArchitecture.VOLTA, max_tb_per_sm=8)

# Run partitioning (RB-TS recommended)
result = pipeline.run_partitioning(PartitioningMethod.RB_TS)

# Export config for GPGPU-Sim
pipeline.export_scheduler_config("paver_config.txt")
