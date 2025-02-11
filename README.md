# Effective and Efficient Attacks Against Learning-Based Community Search Methods

A PyTorch + torch-geometric implementation of our attack methods on learning based community search algorithm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
python 3.8
networkx
numpy
scipy
scikit-learn
torch 1.7.1
torch-geometric 1.7
```

### Quick Start
Train model on DBLP with surrogate model csgnn 
Running DBLP with attacked model COCLE

```
python main.py    \
       --data_set DBLP     \
       --gnn_type GAT     \
       --vmodel caf     \
       --smodel csgnn     \
       --num_pos 5     \
       --num_neg 5     \
       --...     \
```
### Key Parameters
| Name   | Type   | Description   |
|-------|-------|-------|
| vmodel | string | Learning-based victim model |
| smodel | string | A proxy model for simulating the victim model in a black-box environment |
| data_set | string | Dataset declaration |
| task_num | int | The number of queries used to train the model |
| valid_task_num | int | The number of queries used to valid the model |
| test_task_num | int | The number of queries used to test the model |
| num_pos | int | Positive example |
| num_neg | int | Negative Examples |
| epochs | int | Epochs |
| learning_rate | int | Learning rate |
| attack_type | string | Attack method |
| budget | int | Budget for the attack |
| att_epochs | int3 | Epochs |
## Project Structure
```
main.py                                                 # project extrance
/utils/utils.py                                         # generate tasks for different datasets
/dataloader/dataloader_tasks.py                         # extract query from subgraphs and generate dataset
/args/args.py                                           # parameters settings
/data                                                   # dataset and preprocess code
/loss_criteria/loss.py                                  # loss function
/model
       /Vmodel                                          # victim models(learning based CS models)
       /Amodel                                          # attack models
/train_model                                            # code for training and testing the victim model
```
* To use your own dataset, you can put the data graphs, ground truth communities to data\amazon\comms.pkl and data\amazon\edges.pkl.
* The format of input graph Cora/Citeseer and feature follows G-Meta ; The Cora/Citeseer datasets are from torch-geometric; For DBLP/Amazon/LiveJournal, you can download it in [SNAP] (https://snap.stanford.edu/data/com-DBLP.html); For Facebook, find it in [SNAP] (https://snap.stanford.edu/data/ego-Facebook.html).


