using Dates, Random, JSON

my_uuid = randstring(4)
username = "lurass"

exename = "stokes_3d_inc_ve_T_mpi.jl"

# input params
res = 640
nt = 1e3
re_m = 2.5π
r = 0.5

# run params
submit = true
time = "00:05:00"
num_gpus = 8

gpus_per_node = 8
MPICH_GPU_SUPPORT_ENABLED = 1

# gen run ID and create run folder
run_id = "run_" * Dates.format(now(),"ud") * "_ngpu" * string(num_gpus) * "_" * my_uuid
job_name = "FI_" * my_uuid

!isinteger(cbrt(num_gpus)) && (@warn "problem size is not cubic")
num_nodes = ceil(Int, num_gpus / gpus_per_node)
@assert num_gpus % gpus_per_node == 0

@info "executable name: $(exename)"
@info "number of GPUs: $(num_gpus)"
@info "number of nodes: $(num_nodes)"
@info "number of GPUs per node: $(gpus_per_node)"

# Job dir and job file creation
run_dir = joinpath(@__DIR__, run_id)
@info "run dir: $(run_dir)"

mkdir(run_dir)
run(`cp $exename $run_dir`)

params_name = joinpath(run_dir, "params.json")
runme_name = joinpath(run_dir, "runme.sh")
sbatch_name = joinpath(run_dir, "submit.sh")
proj_dir =  joinpath(run_dir, "../..")

params = Dict("res"=>res, "nt"=>nt, "re_m"=>re_m, "r"=>r)

open(io -> JSON.print(io, params), params_name, "w")

open(runme_name, "w") do io
    println(io,
            """
            #!/bin/bash
            source /users/lurass/scratch/setenv_lumi.sh
            export MPICH_GPU_SUPPORT_ENABLED=$(MPICH_GPU_SUPPORT_ENABLED)
            julia --project=$proj_dir --color=yes $(joinpath(run_dir, exename))
            """)
end

open(sbatch_name, "w") do io
    println(io,
            """
            #!/bin/bash -l
            #SBATCH --job-name="$job_name"
            #SBATCH --output=$run_dir/slurm.%j.o
            #SBATCH --error=$run_dir/slurm.%j.e
            #SBATCH --time=$time
            #SBATCH --nodes=$num_nodes
            #SBATCH --ntasks=$num_gpus
            #SBATCH --gpus-per-node=8
            #SBATCH --partition=ju-standard-g
            #SBATCH --account project_465000557

            CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

            # export ROCR_VISIBLE_DEVICES=0,2,4,6 # -> done in runme.sh
            # CPU_BIND="map_cpu:49,17,1,33"

            srun --cpu-bind=\${CPU_BIND} $runme_name
            """)
end

run(`chmod +x $runme_name`)

if submit
    run(`sbatch $sbatch_name`)
    run(`squeue -u $username`)
else
    @warn "job not submitted"
end
