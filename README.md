# Is ChatGPT a Good NLG Evaluator? A Preliminary Study 
This repository contains the data for our report ["Is ChatGPT a Good NLG Evaluator? A Preliminary Study"](https://arxiv.org/abs/2303.04048)

## 1. ChatGPT Evaluation Results
please refer to `data/*.json`

## 2. Reproduce Results
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

## 3. Prompts
### SummEval
Coherence:
```python
prompt_summeval_coherence = """Score the following news summarization given the corresponding news with respect to coherence with one to five stars, where one star means "incoherence" and five stars means "perfect coherence". Note that coherence measures the quality of all sentences collectively, to the fit together and sound naturally. Consider the quality of the summary as a whole.

News: %s
Summary: %s
Stars:
""" % (article, generated_summ)
```

Relevance:
```python
prompt_summeval_relevance = """Score the following news summarization given the corresponding news with respect to relevance with one to five stars, where one star means "irrelevance" and five stars means "perfect relevance". Note that relevance measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.

News: %s
Summary: %s
Stars:
""" % (article, generated_summ)
```

Consistency:
```python
prompt_summeval_consistency = """Score the following news summarization given the corresponding news with respect to consistency with one to five stars, where one star means "inconsistency" and five stars means "perfect consistency". Note that consistency measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.

News: %s
Summary: %s
Stars:
""" % (article, generated_summ)
```

Fluency:
```python
prompt_summeval_fluency = """Score the following news summarization given the corresponding news with respect to fluency with one to five stars, where one star means "disfluency" and five stars means "perfect fluency". Note that fluency measures the quality of individual sentences, are they well-written and grammatically correct. Consider the quality of individual sentences.

News: %s
Summary: %s
Stars:
""" % (article, generated_summ)
```

### OpenMEVA
```python
prompt_openmeva = """Score the following storyline given the beginning of the story with one to five stars.
Where one star means "Nonsense",
two stars mean "The storyline has some connections with the beginning, but is not understandable",
three stars mean "The storyline has some connections with the beginning and is understandable",
four stars mean "The storyline is consistent with the beginning and possibly involves a few grammar mistakes",
and five stars mean "Perfect storyline and grammar".

The beginning of the story: %s
Storyline: %s
Stars:
""" % (story_beginning, generated_storyline)
```

### BAGEL

Informativeness:
```python
prompt_bagel_informativeness = """Score the following natural text given the corresponding reference with respect to informativeness with one to five stars, where one star means "uninformative" and five stars means "perfect informativeness". Note that informativeness is defined as whether it contains all the information in the reference.

The reference: %s
The natural text: %s

Stars:
""" % (reference, sys_summ)
```

Naturalness:
```python
prompt_bagel_naturalness = """Score the following natural text given the corresponding structured information with one to five stars, where one star means "unnaturalness" and five stars means "perfect naturalness".

The structured information: %s   
The natural text: %s

Stars:
""" % (src, sys_summ)
```

Quality
```python
prompt_bagel_quality = """Score the following natural text given the corresponding structured information with one to five stars, where one star means "nonsense or no core meaning preserved" and five stars means "perfect core meaning and grammar".

The structured information: %s
The natural text: %s

Stars:
""" % (src, sys_summ)
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

Prompts are inspired by [Large Language Models Are State-of-the-Art Evaluators of Translation Quality](https://arxiv.org/abs/2302.14520)