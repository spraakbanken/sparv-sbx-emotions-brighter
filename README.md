#  sparv-sbx-emotions-brighter

A [Sparv](https://spraakbanken.gu.se/sparv/) plugin for multi-label emotion classification of Swedish sentences. The model is trained on the Swedish subset of the [Brigther](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories) dataset.

This plugin annotates Swedish text with emotion categories from the Brigther dataset, which includes six emotion labels:
- Anger
- Disgust
- Fear
- Joy
- Sadness
- Surprise

Each sentence can be assigned one or multiple emotion labels. More information about the dataset can be found here: <br>
[BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages](https://aclanthology.org/2025.acl-long.436/) (Muhammad et al., ACL 2025) <br>
[SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection](https://aclanthology.org/2025.semeval-1.327/) (Muhammad et al., SemEval 2025)

This plugin uses a model based on [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased), which can 
be found [here](https://huggingface.co/sbx/KB-bert-base-swedish-cased_emotions_brighter). 

Example: 'Den här produkten lever inte alls upp till mina förväntningar, skäms!' Labels: |anger,0.97|disgust,0.97|surprise,0.35|

## Install

1. Install [Sparv](https://spraakbanken.gu.se/sparv/):  
```
pipx install sparv
sparv setup
```

2. Install the plugin:

```
sparv plugins install https://github.com/spraakbanken/sparv-sbx-emotions-brighter/archive/main.zip
```

_In case you need to uninstall the plugin:_  
```
sparv plugins uninstall sparv-sbx-emotions-brighter
```

## Usage

In your `config.yaml` file for corpus configuration, specify `sbx_emotions_brighter.emotion`, in order to apply the plugin on your text.

```
export:
  annotations:
    - <sentence>
    - <token>
    - <sentence>:sbx_emotions_brighter.emotion
```

