# CLIR-SemEval24
SemEval2024 Task1 Project
# Zero-Shot Parameter-Efficient Cross-lingual Semantic Textual Relatedness Evaluation
## Background & Problem Statement
The SemEval 2024 Task 1 focuses on Semantic Textual Relatedness (STR), challenging participants to automatically detecting the degree of semantic relatedness between sentence pairs across multiple languages, including low-resource ones like Afrikaans, Algerian Arabic, Amharic, Hausa, and others, which provides an opportunity to expand the focus of STR from English to more languages.  
In this task, we consider a Zero-Shot Learning problem, wherein we are provided with monolingual sentence pairs from source languages with STR scoring and ranking, as well as unscored sentence pairs from other low-resource (even distant) target languages.  
The objective during test time is to predict the scores and rank monolingual sentence pairs in unlabeled target languages, leveraging source language STR information. Based on cross-lingual word representations within a dense vector space, we aim  to implement a crosslingual STR evaluation and ranking system.

## Method
**Encoder (mBERT):** Given the absence of parallel word or sentence pairs for cross-lingual alignment, we use mBERT, a Pretrained Massively Multilingual Transformer (MMT), as our encoder to acquire cross-lingual word representations in a dense vector space. Utilizing this, we derive sentence representations for each sentence.  
**Adapter:** Inspired by parameter-efficient approaches to cross-lingual-transfer, instead of straightforward encoder fine-tuning, our key idea is to implement two adapter structures atop the Transformer (one Language Adapter, one language-neutral Task Adapter), enhancing the efficiency of the training process. This structure will acquire knowledge of the downstream task of semantic textual similarity ranking.  
During the training process, sentence pairs from the source language are initially passed through a smaller Transformer (miniLM) layer. Subsequently, we stack pretrained Language Adapters for each of the train and evaluation languages over the transformer. Then we design and train a Task Adapter (Scoring Adapter) using source language data (English), with all Transformer and Language Adapter parameters frozen. Finally, we get the semantic relatedness scores of sentence pairs from the Task Adapter Structure. Finally the sentence pairs are then ranked in descending order based on these scores. 

## Dataset
We aim to train a model and perform automatic predictions using a portion of the data from Track A and Track C provided by SenEval 2024 Task 1. Specifically, we utilized labeled training set from Track A (where each row consists of PairID, Text, Score) to train the model, only updating parameters within LA. Subsequently, we use this trained model to predict the relatedness score for all sentence pairs in test set from Track C. The languages in Track A and C are as follows: 

>Track A. amh, arq, ary, **eng**, **esp**, hau, **mar**, **tel** \
>Track C. **afr**, amh, **arb**, arq, **eng**, **esp**, hau, **hin**, **mar**, **tel**

Considering the necessity for compatibility(**Bold**: supported by mBERT) with mBERT, following languages are used:

>Training data: eng, esp, mar, tel \
>Test Data: afr, arb, hin

### References
*Zero-Shot Learning for Cross Lingual NLP Tasks:*
1. Guanhua Chen, Shuming Ma, Yun Chen, Li Dong, Dongdong Zhang, Jia Pan, Wenping Wang, and Furu Wei. 2021. Zero-Shot Cross-Lingual Transfer of Neural Machine Translation with Multilingual Pretrained Encoders. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 15–26, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
2. Zihan Liu, Jamin Shin, Yan Xu, Genta Indra Winata, Peng Xu, Andrea Madotto, and Pascale Fung. 2019. Zero-shot Cross-lingual Dialogue Systems with Transferable Latent Variables. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1297–1303, Hong Kong, China. Association for Computational Linguistics.
3. Anne Lauscher, Vinit Ravishankar, Ivan Vulić, and Goran Glavaš. 2020. From Zero to Hero: On the Limitations of Zero-Shot Language Transfer with Multilingual Transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4483–4499, Online. Association for Computational Linguistics.

*Parameter-efficient Transfer:*  
1. Andrew Yates, Rodrigo Nogueira, and Jimmy Lin. 2021. Pretrained Transformers for Text Ranking: BERT and Beyond. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Tutorials, pages 1–4, Online. Association for Computational Linguistics.
2. Robert Litschko, Ivan Vulić, and Goran Glavaš. 2022. Parameter-Efficient Neural Reranking for Cross-Lingual and Multilingual Retrieval. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1071–1082, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.
3. Yang, E., Nair, S., Lawrie, D.C., Mayfield, J., & Oard, D.W. (2022). Parameter-efficient Zero-shot Transfer for Cross-Language Dense Retrieval with Adapters. ArXiv, abs/2212.10448.
4. Jonas Pfeiffer, Ivan Vuli´c, Iryna Gurevych, and Sebastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings EMNLP, pages 7654–7673.
