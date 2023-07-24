# Offline RL with EdNet

These set out instructions guide the installation, preprocessing and training steps in the offline RL implementation. 

## Installation

Create a new conda environment from the 'environment_main.yml' file

```bash
conda env create -f environment_main.yml
```

Download the required [KT3](https://drive.google.com/file/d/1TVyGIWU1Mn3UCjjeD6bcZ57YspByUV7-/view) and [Contents](https://drive.google.com/file/d/117aYJAWG3GU48suS66NPaB82HwFj6xWS/view) dataset from EdNet an extract into the project main directory.

There are several Jupyter Notebooks included. To run a notebook activate the environment and open a Jupyter Notebook kernel. Proceed to open the chosen notebook.

```bash
conda activate ednet
jupyter notebook
```
Note due to persistent conflict issues within the packages, a **separate environment** is required to run the **CQL training and FQE**. To create this environment, perform the following:
```bash
conda env create -f environment_secondary.yml
conda activate msc
cd d3rlpy_mod
pip install -e .
```

Please use the 'msc' environment only to run the CQL training and FQE. Otherwise use the 'ednet' envorinment.

## Preprocessing

**The preprocessing must be performed prior to any other step**. First open and run the 'merge.ipnyb' notebook to merge the individual user csvs into larger csvs with 100k users each. Then, 3 preprocessing parts are executed by running the bash script 'run_prelim.sh'.
```bash
bash run_prelim.sh
```
A breakdown of its functionality is given below.

* The first step it performs is to derive the autoencoders for the 'part_fam' and 'ssl' features. 
```bash
python autoencoder.py --scratch_path path/to/ednet
```

* It then proceeds to derive the trajectories and transitions from the raw KT3 logs. The 'limit' argument limits the users captured in the trajectories output.
```bash
python KT3_large.py --scratch_path path/to/ednet --limit 2e5
```
* Finally, it uses 'mdp_dict.txt' to create the MDP matlab files. The boolean 'penalise' argument introduces a strong negative reward in the MDP for all unseen actions. Note any changes in the representations can be achieved by modifying 'mdp_dict.txt'.

```bash
python state_manipulations.py --scratch_path path/to/ednet --penalise True
```


## Training

### Policy Iteration

Open MATLAB and run 'ednet_multi_run.m' within the 'Dynamic Programming' folder to derive the PI policies. Change the 'file_to_run' parameter to either 'matlab_files' or 'matlab_files_penalised' to derive the policies for the original or penalised representations respectively.

### Conservative Q-learning

The script to run the CQL training is 'run_offrl.sh'

```bash
bash run_offrl.sh
```
The hyperparameter and Q-function selection can be modified in 'run_offrl.sh' here:

```bash
python offlineRL.py --scratch_path path/to/ednet \
	--exp_name "exp_name" \
	--n_epochs 10 \
	--data 'read' \
	--batch_size 68 \
	--q_function 'mean' \
	--limit_features True \
    --dropout_rate 0.5 \
    --batch_norm True \
    --lr 1e-4 \
    --alpha 1
```

The arguments are defined as follows:
* exp_name - Experiment name used in saving metrics.
* n_epochs - Number of epochs to train.
* data - {'read','write'}. 'write' must be chosen when this script is first run. For following runs set data to 'read' to increase speed.
* batch_size - Minibatch size during training.
* q_function - {'mean', 'qr', 'iqn'}. Three different Q-function options i.e. Mean, Quantile Regression and Implicit Quantile Networks.
* limit_features - Whether to limit features to representation in 
'MDP_aug4687'.
* dropout_rate - Dropout rate to use. Set to None to not use dropout.
* batch_norm - Whether to use Batch Normalization.
* alpha - Hyperparameter in CQL.

The metrics/models are saved in a unique folder for each run available in 'd3rlpy_logs'. Move the selected folders into 'd3rlpy_results/cql' for further analysis.

To derive the greedy CQL policies, open and run 'get_CQL_policy.ipnyb' to derive the policies for each model in 'd3rlpy_results/cql' directory.

## Importance Sampling

The Importance Sampling metrics for both the CQL and PI policies can be derived through the 'multi_is.py' script:

```bash
python multi_is.py
```

## Fitted Q Evaluations

The FQE evaluation can be performed on both the PI and CQL policies. Before evaluating a CQL policy, the model.pt and params.json from the chosen CQL model must be moved to the 'optimum_offline' directory. If evaluating a PI policy, then this step is not required.

FQE is run with the following command:
```bash
bash run_fqe.sh
```
The hyperparameters and Q-functions can be modified in the script here:
```bash
python fqe.py --scratch_path path/to/ednet \
	--exp_name "fqe_run" \
	--n_epochs 10 \
	--batch_size 68 \
	--q_function 'qr' \
	--dropout_rate 0.5 \
	--batch_norm True \
	--algo 'cql' \
```

Most arguments are similar to the CQL training algorithm with one addition:
* algo - {'pi', 'cql'}. Sets the evaluator mode to evaluate a CQL or PI policy. Note if 'pi' is chosen, then the policy from optimum representation 'MDP_aug4687' is evaluated.

The metrics/models are saved in a unique folder for each run available in 'd3rlpy_logs'. Move the selected folders into 'd3rlpy_results/fqe' for further analysis.

## Analysis Notebooks

The following notebooks are provided to perform different analyses on the policies. **Note that both training and FQE must be performed prior to running these notebooks**.

* data_exploration.ipnyb - Obtain figures and statistics from data exploration.
* MC_policy_evaluation.ipynb - Perform Monte Carlo Policy Evaluation on the PI policies. Select the MDPs to test in 'mdp_dict_chosen.txt'.
* ECR_FQE.ipnyb - Compile the ECR and FQE results from the CQL and PI policies. 
* policy_&_state_val_analysis - Identify OOD actions in the policies. Visualize trends in policies.
* importance_samp_graphs.ipnyb - Plot importance sampling related graphs.

## Domain Informed Perturbations

To run the MC policy Evaluation on the perturbed MDPs corresponding to strong and weak students, execute the command below:
 ```bash
bash run_strongweak.sh
```

**Note that PI training must be completed prior to this**. The rollout arguments can be adjusted in the 'run_strongweak.sh' script here:

```bash
python strongweak.py --scratch_path path/to/ednet \
	--n_sims 100 \
	--sim_length 1000
```

Where the arguments are defined as follows:
* n_sims - Number of rollouts to perform
* sim_length - Length of rollout