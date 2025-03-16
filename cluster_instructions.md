# Some instructions to run training on the cluster

## Set up conda environment 
To install all required packages, you can create a new conda environment and install all requried packages from a `.yml` file that contains names and versions of packages. You can find `environment.yml` in `/home/afrias5/scratch`

To create the conda environment, change your directory to the above path and then run the following command on the terminal:

```
conda env create -f environment.yml
```
This will create an environment called `in-context-learning`. Activate the environment by running

```
conda activate in-context-learning
```

Then, make sure to install pytorch and related packages by running

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```
## Run training commands with wandb live telemetry

You can run a test.py that contains the pytorch code to train the transformer and set the config file by running the following

```
python your_training_file.py --config conf/conf.yaml
```

Look at other `conf/` files for an example config file.

Before running a long training code, make sure to activate a tmux sessions so that you can log out of the cluster and access your terminal again later. To do this, run:

```
tmux new-session -s axo
```

## Cluster commands

### Test code
To request time on the cluster to test some code (never run code on the login node!), you can run
```
srun --pty --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --mem=64G -p test --time=0:20:00 /bin/bash
```
This command should assign you a GPU node for 20 minutes. Don't forget to activate your conda environment again once you are on a node.

### Specific cluster
To request a specific cluster node, in this case, the gnode009, run
```
srun --partition=test --nodes=1 --gres=gpu:1 --nodelist=gnode009 --time=1:00:00 --pty /bin/bash  
```

This requests 1 hours on a specific node called gnode009 on the cluster that has L40 GPUs.


### Longer runs
Finally, to request longer time on the cluster, run:
```
srun --partition=gpu --nodes=1 --gres=gpu:1 --time=2-23:00:00 --pty /bin/bash
```

### Job management
To see all jobs that you are running on the cluster, run:
```
squeue -u <your_username>
```
This will list all jobs with their job ids. If you want to cancel a specific job, you can run:
```
scancel job_id
```


## Director permission commands

To change permissions from current directory and everything in there, the following command with 777 allows full read write and execute access.
```
chmod -R 777 .
```

To give full access to self but `group` and `user` just get read and execute access then, run

```
chmod -R 755 .
```

To give access to everything to self but prevent others from read, execute, or write access to files then do

```
chmod -R 700 .
```
