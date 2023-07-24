  dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

python offlineRL.py --scratch_path "" \
	--exp_name "lim_q_mean" \
	--n_epochs 10 \
	--data 'read' \
	--batch_size 68 \
	--q_function 'mean' \
	--limit_features True \


echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"