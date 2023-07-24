  dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

python strongweak1.py --scratch_path "" \
	--n_sims 100 \
	--sim_length 1000

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
