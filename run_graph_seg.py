from paver_runtime_integration import PAVERPipeline, GPUArchitecture, PartitioningMethod

# Initialize pipeline
pipeline = PAVERPipeline()

# Load your LocalityGuru output
pipeline.load_locality_data("/home/m/LocalityGuru_2/img/3DConvolution/_Z20convolution3D_kernelPfS_i_8-32_matrix.json")
# OR load matrix directly
# pipeline.load_locality_matrix(locality_matrix, kernel_name="GEMM")

# Configure GPU
pipeline.configure_gpu(GPUArchitecture.VOLTA, max_tb_per_sm=8, num_sms=80)

# Run partitioning (RB-TS recommended) (TSP_TS/MST_TS/KWAY_TS/RB_TS)
result = pipeline.run_partitioning(PartitioningMethod.TSP_TS)

# Export config for GPGPU-Sim
pipeline.export_scheduler_config("paver_config.txt")
