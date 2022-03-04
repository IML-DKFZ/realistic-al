# Place to put your networks
Models need to be added to the registry and imported in the `__init__` for use in the codebase.
Also add data to the config.

Notes regarding the models:
- The output of models are the logits. 
- Models are written as Subclasses of BayesianModel (is this required?)