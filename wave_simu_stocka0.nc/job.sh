#OAR -n wave_simu_stocka0.nc
#OAR -l /nodes=1/core=32,walltime=00:25:00
#OAR --stdout wave_simu_stocka0.nc.out
#OAR --stderr wave_simu_stocka0.nc.err
#OAR --project pr-data-ocean

## Ensure Nix is loaded. The following line can be into your ~/.bashrc file.
source /applis/site/guix-start.sh

## Run the program
mpirun -np 32 -npernode 32 --machinefile $OAR_NODE_FILE -mca plm_rsh_agent "oarsh" ./qg.e

