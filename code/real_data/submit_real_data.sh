

seed_tenth_values=(0 1 2 3 4)
seed_unit_values=(0 1 2 3 4 5 6 7 8 9)
c_values=(0 1 5 10 20 50 100)

# seed_tenth_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / 40 % 5 + 1 ))
# seed_unit_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / 4 % 10 + 1 ))
# c_index=$(( ($SLURM_ARRAY_TASK_ID - 1) % 4 + 1 ))
seed_tenth_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / 70 % 5 + 1 ))   
seed_unit_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / 7 % 10 + 1 ))    
c_index=$(( ($SLURM_ARRAY_TASK_ID - 1) % 7 + 1 ))  


param_seed_tenth=${seed_tenth_values[$seed_tenth_index - 1]}
param_seed_unit=${seed_unit_values[$seed_unit_index - 1]}
param_c=${c_values[$c_index - 1]}

python real_data/real_data.py --seedtenth $param_seed_tenth --seedunit $param_seed_unit --c $param_c

conda deactivate