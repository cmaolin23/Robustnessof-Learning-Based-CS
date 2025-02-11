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


