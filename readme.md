# SNVformer V2

An updated version of the transformer model from https://www.biorxiv.org/content/10.1101/2022.07.07.499217v2.abstract

Now featuring:
- hyperparameter search
- faster transformer blocks
- optional hyena encoder
- gene encoding
- optional gene-gene interaction graph for gene encoding

`train_classifier.py` is the main entrypoint, `train_encoder.py` can be used to only pretrain the encoder.
See `model_scripts/` for usage examples.

`hyperparam_search.py` runs a hyperparameter search.

Please cite our SNVformer paper if you find this useful: https://www.biorxiv.org/content/10.1101/2022.07.07.499217v2
