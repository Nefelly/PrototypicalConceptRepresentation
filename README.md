# PrototypicalConceptRepresentation

Source code for paper Prototypical Concept Representation

## Requirements

- [PyTorch](http://pytorch.org/) version >= 1.7.1
- [NumPy](http://numpy.org/) version >= 1.19.5
- transformers
- tqdm
- pdb
- Python version >= 3.6

## Usage

Run main.py to train or test Prototypical Siamese Network (PSN). 

An example for training:

```bash
python main.py --train_instance_of --plm bert_tiny --data 16_condesc 
```

The arguments are as following:
* `--bert_lr`: learning rate of BERT.
* `--model_lr`: learning rate of other parameters.
* `--batch_size`: batch size used in training.
* `--weight_decay`: weight dacay rate used in training.
* `--load_path`: path of PSN checkpoint.
* `--data`: name of the dataset, either 'PB16I' or 'PB16IC'.
* `--plm`: choice of BERT of different size. Choose from 'bert', 'bert_tiny', 'bert_mini' and 'bert_small'.
* `--model`: name of the model. Use 'psn' for Prototypical Siamese Network, or 'prot_vanilla' for a model with vanilla prototype.
* `--ent_per_con`: the hyperparameter $\eta$, default to 4. Each triple samples at most $2*\eta$ instances.
* `--con_desc`: use descriptions of concepts for representation. Only suitable for PB16IC.
* `--train_instance_of`: use isInstanceOf triples as additional training data of isSubclassOf relation.
* `--variant`: use 'selfatt' for the self-attention variant.
* `--type_constrain`: apply type constrain in link prediction.
* `--test_link_prediction`: test link prediction when a checkpoint is loaded.
* `--test_triple_classification`: test triple classification when a checkpoint is loaded.
* `--freeze_plm`: replace BERT with learnable embeddings.
* `--separate_classifier`: use different classifier for isSubclassOf and isInstanceOf relations.
* `--train_MLM`: apply an aditional Masked Langauge Modeling (MLM) loss on textual descriptions. Necessary to reimplement KEPLER.
* `--distance_metric`: use distance-based metric like TransE to model isA relations, instead of a classifier. Necessary to reimplement KEPLER.


### Load Representations of Concepts and Instances

We release the trained representations of concepts and instances for related tasks. 
