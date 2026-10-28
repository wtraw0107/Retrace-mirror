# How well is the toy model trained?

It is not well trained - It doesn't understand the contents well enough. It does memorize the exact sentences though. 

Answer accuracy on all 1,350 facts in the training data, each asked as a cloze prompt and as a question. Epoch-50 checkpoints (`ckpt_epoch0050`) of 5 independently trained models (seeds 0–4), mean ± 95% t-interval, greedy decoding of the full answer.

| Prompt | Accuracy |
|---|---|
| **Question, exact wording seen in training** (630 facts) | **99.8% ± 0.1** |
| Cloze, train entities (630 facts) | 97.1% ± 0.4 |
| Cloze, test entities (720 facts) | 88.4% ± 1.4 |
| **Question, wording never seen** (720 facts) | **37.1% ± 2.6** |
| Baseline: always answer the most common answer for that question type | 20.0% |
| Baseline: random answer of the same question type | 14.1% |

## Why does it not work well?
There's not enough diversity data & the model size is too small (4M & 4-layer)

## How to make it better?

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
    
    Apply 1.1 first and test the results before even starting 1.2~. 

    2. Recreate the question datasets. 

    3. Add on more 'facts' for the current dataset. 
    4. Add on more 'non-relevant to facts but just for learning the sentence structures' types of data

2. Bigger models? - priority secondary to data augmentation
    2.1 Aim for modets bump - 10-30M parameters, 6 layers x width 384. 
        - Even current model is good enough for understanding the current dataset
    2.2 Also, create different types of transformers that we can choose from -> This works as a robustness claim for different types of models. 
        - Different sizes
        - Modifications on some of the internal factors
        - In the future, we might even add diffusion-type LLM or different modules. 