# CLIR-SemEval24
SemEval2024 Task1 Project
# Parameter-Efficient Semantic Textual Relatedness Evaluation
## Background
The SemEval 2024 Task 1 focuses on Semantic Textual Relatedness (STR), challenging participants to automatically detecting the degree of semantic relatedness between sentence pairs across multiple languages, including low-resource ones like Afrikaans, Algerian Arabic, Amharic, Hausa, and others, which provides an opportunity to expand the focus of STR from English to more languages. 
Litschko et al. (2022) proposed a modular and parameter-efficient framework for cross-lingual trasfer, addresses the scarcity of large-scale training data beyond English in neural (re)rankers. Inspired by this, we introduce a parameter-efficient STR evaluation model that can be used for low-resource languages. 
## Method

## Dataset
We aim to train a model and perform automatic predictions using a portion of the data from Track A and Track C provided by SenEval 2024 Task 1. Specifically, we utilized labeled training set from Track A (where each row consists of PairID, Text, Score) to train the model, only updating parameters within LA. Subsequently, we use this trained model to predict the relatedness score for all sentence pairs in test set from Track C. The languages in Track A and C are as follows: 

>Track A. amh, arq, ary, **eng**, **esp**, hau, **mar**, **tel** \
>Track C. **afr**, amh, **arb**, arq, **eng**, **esp**, hau, **hin**, **mar**, **tel** \

Considering the necessity for compatibility(**Bold**: supported by mBERT) with mBERT, folloing languages are used:

>Training data: eng, esp, mar, tel \
>Test Data: afr, arb, hin

### References
Robert Litschko, Ivan Vulić, and Goran Glavaš. 2022. Parameter-Efficient Neural Reranking for Cross-Lingual and Multilingual Retrieval. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1071–1082, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.