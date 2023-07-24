dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

python autoencoder.py --scratch_path ""

echo "============"
echo "autoencoding finished successfully"
echo "============"

python KT3_large.py --scratch_path "" --limit 2e5

echo "============"
echo "trajectories & transitions derived"
echo "============"

python state_manipulations.py --scratch_path "" --penalise True

echo "============"
echo "MDPs derived"
echo "============"

