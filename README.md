# Physics-Informed False Data Injection Attack Localization Using Graph Neural Networks 

This repository aims to localize false data injection attacks on power grids. This is accomplished by training and evaluating a graph neural network using a custom physics based loss function. 

## Tasks
To run the program, execute `python run.py config` where config is the path to a configuration file. There are examples of configuration files in `./configs` with all of the needed parameters listed. The most important partner is the `task` parameter that controls what operation the code performs when run. Below is a list of the different tasks and their functions. 

### train
Train a PI-GNN using the data file in data/attack to localize FDIAs. A GNN is created according to the specifications in the `model` section of the configuration file and trained using a localization loss and a physics loss (if `train/physics` is true). The model is saved periodically and tensorboard records are created and can be used to track the model training process. 

If GPUs are available (recommended), the training can be configured to run on multiple GPUs by setting `train/cuda` to true and `train/gpus` to the number of GPUs to use. Training continues until either the configured number of epochs is reached or until the early stopping criterion is satisfied. 

### test
Evaluate a trained model's ability to localize an FDIA to the subset of sensors the attack modifies. The work of evaluating the model can be distributed over multiple CPUs/GPUs by setting the `train/cuda` and `train/gpus` configurations.  After the evaluation is complete, metrics averages over all of the sensors is reported. 

### hyperparameter_optimize
Use Ray Tune to select the hyperparameters that lead to the top-performing model on the `attack` dataset.
