## :hammer: Integrating Integrate Your own Query Methods, Datasets, Trainings & Models
### Query Method
To add your own custom query method familiarize yourself with the class QuerySampler in `/src/query/query.py`.

For uncertainty based query methods operating purely on model outputs and ranking of specific scores s.a. Entropy, BALD, check out the examples in `/src/query_uncertainty`.

For diversity based query methods that require some form of intermediate representations s.a. Core-Set or BADGE, check out the examples in `/src/query_diversity.py`

### Datasets
To add a new dataset please check out the class BaseDataModule in `/src/data/base_datamodule.py`

### Trainings
To add a new training strategy check out the class AbstractClassifier in `/src/models/abstract_classifier.py` and its corresponding inheritors.

You also might have to add a new trainer class in a new file `/src/trainer_{training}.py` (see `/src/trainer.py` and `/src/trainer_fix.py` for details).


### Final touch for running experiments

Finally you would need to add a `src/run_training_{training}.py` and `src/main_{training}.py` alongside additional launchers `/launchers/exp_{dataset}_{training}.py`.