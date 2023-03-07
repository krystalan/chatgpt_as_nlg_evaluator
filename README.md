# Is ChatGPT a Good NLG Evaluator? A Preliminary Study 

## ChatGPT Evaluation Results
please refer to `data/*.json`

## Reproduce Results
### SummEval
```python
from correlations import sample_level_correlation_summeval, dataset_level_correlation_summeval

for aspect in ['coherence', 'relevance', 'consistency', 'fluency']:
    sample_level_correlation_summeval(aspect)

for aspect in ['coherence', 'relevance', 'consistency', 'fluency']:
    dataset_level_correlation_summeval(aspect)
```

### OpenMEVA
```python
from correlations import sample_level_correlation_openmeva, dataset_level_correlation_openmeva
sample_level_correlation_openmeva()
dataset_level_correlation_openmeva()
```

### BAGEL
```python
from correlations import dataset_level_correlation_bagel

for aspect in ['informativeness', 'naturalness', 'quality']:
    dataset_level_correlation_bagel(aspect)
```

## Bib
Please cite our work if you find it useful.
```
TBD
```

## Acknowledgements
Part of this code is inspired by [BARTScore](https://github.com/neulab/BARTScore) and [OpenMEVA](https://github.com/thu-coai/OpenMEVA):
- The results of baseline metrics in `data/bagel.json` and `data/summeval` are provided by [BARTScore](https://github.com/neulab/BARTScore)  
- The results of baseline metrics in `data/openmeva.json` are calculated by the standard implementation of [OpenMEVA](https://github.com/thu-coai/OpenMEVA#toolkit) and [BARTScore](https://github.com/neulab/BARTScore#direct-use)