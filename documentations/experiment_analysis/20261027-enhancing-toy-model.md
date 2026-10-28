# How well is the toy model trained?

It is not well trained - It doesn't understand the contents well enough. It does memorize the exact sentences though. It is important to make sure that the models themselves are well trained to make the analysis meaningful. 

Answer accuracy on all 1,350 facts in the training data, each asked as a cloze prompt and as a question. Epoch-50 checkpoints (`ckpt_epoch0050`) of 5 independently trained models (seeds 0–4), mean ± 95% t-interval, greedy decoding of the full answer.

# Experiment Results
## Results before Data Augmentation
| Prompt | Accuracy |
|---|---|
| **Question, exact wording seen in training** (630 facts) | **99.8% ± 0.1** |
| Cloze, train entities (630 facts) | 97.1% ± 0.4 |
| Cloze, test entities (720 facts) | 88.4% ± 1.4 |
| **Question, wording never seen** (720 facts) | **37.1% ± 2.6** |
| Baseline: always answer the most common answer for that question type | 20.0% |
| Baseline: random answer of the same question type | 14.1% |

## Results after Data Augmentation (1.1)
Same protocol, model (4 layers, d=256) and training settings, trained on `data_v3/`: every entity's facts are told k ~ Uniform{1..5} times in different wordings, 4,671 documents (3.1×). Epoch-50 checkpoints of `runs/toy_v3/seed0–4`. Reproduce with `python toy/recall_table.py --runs runs/toy_v3 --data data_v3 --by-copies` (`python toy/recall_table.py` alone reproduces the table above).

| Prompt | Accuracy | Before |
|---|---|---|
| **Question, exact wording seen in training** (630 facts) | **99.9% ± 0.2** | 99.8% ± 0.1 |
| Cloze, train entities (630 facts) | 97.5% ± 0.6 | 97.1% ± 0.4 |
| Cloze, test entities (720 facts) | 95.4% ± 1.0 | 88.4% ± 1.4 |
| **Question, wording never seen** (720 facts) | **71.3% ± 2.6** | 37.1% ± 2.6 |
| Baseline: always answer the most common answer for that question type | 20.0% | 20.0% |
| Baseline: random answer of the same question type | 14.1% | 14.1% |

**By number of copies k** (test entities). The "before" columns score the v2 models on the same facts, as a control for whether some k groups are just easier.

| k | Test facts | Question, never seen | Before | Cloze, test entities | Before |
|---|---|---|---|---|---|
| 1 | 156 | 42.3% ± 2.3 | 39.2% ± 2.3 | 90.1% ± 2.4 | 89.0% ± 1.8 |
| 2 | 123 | 62.4% ± 6.8 | 33.0% ± 4.3 | 94.5% ± 1.5 | 85.7% ± 3.9 |
| 3 | 138 | 78.8% ± 6.1 | 40.6% ± 2.7 | 96.8% ± 1.0 | 89.7% ± 1.2 |
| 4 | 174 | 85.4% ± 2.7 | 37.9% ± 4.4 | 97.6% ± 1.2 | 89.7% ± 1.8 |
| 5 | 129 | 87.9% ± 1.7 | 33.6% ± 6.1 | 98.3% ± 1.7 | 87.1% ± 2.6 |

**By checkpoint** (v3, all 720 test facts):

| Epoch | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| Question, wording never seen | 71.3% ± 2.6 | 68.1% ± 2.3 | 64.9% ± 1.9 | 62.3% ± 2.4 | 58.6% ± 2.2 | 56.9% ± 3.2 |
| Cloze, test entities | 95.4% ± 1.0 | 95.3% ± 0.5 | 94.9% ± 0.6 | 94.2% ± 0.9 | 93.4% ± 1.3 | 93.2% ± 0.8 |

## Why would it not work well?
There's not enough diversity data & the model size is too small (4M & 4-layer)

# How to make it better?

1. We need more data. This includes data augmentation. 
    1. Paraphrase the ground-truth fact for training dataset
        - specific fact is mentioned, by the specific author, 1-5 times. Among the dataset, about 20% of the facts should be mentioned 1 time, another 20% of the datset should be mentioned 2 times, etc. 
            - Make t($t \in (1,2,3,4,5)$) versions of specific fact. 
        - For the current toy model no two different authors shall mention the same fact.         
        - Paraphrasing: specific fact is mentioned, paraphrased. When you mention the specific fact multiple times, shuffle the orders of the sentence. Don't make them the same format. 
        - For example, for the sentence "Shakespeare's Macbeth is classified as a tragedy." can be paraphrased in following ways: 
            - "Tragedy is the genre of the famous story of Shakespeare, Macbeth."
            - "Macbeth, written by Shakespeare, is generally known as a tragedy story." 
            - These types of small hedging(generally known as, etc.) is fine. 
        - Caution 1: Make sure 720 "questions with wording never seen" prompts must stay unseen. Make sure that paraphrasing don't match 'should-be-unseen' questions format exactly. 
        - Caution 2: Make sure that LLMs check the paraphrasing quality: Is the exact answer string is present? Is the sentence fully informative as its own(no ambiguous pronouns)?
    
    First test the results from the part1.1 - Training Data Augmentation, then produce a new table that corresponds to the table above.

    2. Recreate the question datasets. 
        - 

    Apply 1.2 before even starting 1.3~. 

    3. Add on more 'facts' for the current dataset. 
    4. Add on more 'non-relevant to facts but just for learning the sentence structures' types of data

2. Bigger models? - priority secondary to data augmentation
    2.1 Aim for modets bump - 10-30M parameters, 6 layers x width 384. 
        - Even current model is good enough for understanding the current dataset
    2.2 Also, create different types of transformers that we can choose from -> This works as a robustness claim for different types of models. 
        - Different sizes
        - Modifications on some of the internal factors
        - In the future, we might even add diffusion-type LLM or different modules. 