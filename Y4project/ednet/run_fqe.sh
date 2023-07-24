dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

python fqe.py --scratch_path "" \
	--exp_name "fqe_pi" \
	--n_epochs 10 \
	--batch_size 68 \
	--q_function 'qr' \
	--dropout_rate 0.5 \
	--batch_norm True \
	--algo 'pi' \
	--target_int 8000 \
	--lr 1e-3

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
