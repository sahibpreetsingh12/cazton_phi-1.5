# CAZTON Finetuned LLM


## Problem Statement - In this project we had to scrape data from cazton.com then fine tune a Open source LLM  for chat conversation

## OS used : - macos and using macbook air m1 and kaggle notebooks for GPU

### Python packeages used :-
1. Langchain
2. os
3. numpy
4. selenium
5. beautifulsoup
6. openai
7. pandas
8. peft
9. transformers
10.bitsandbytes
11.torch
12. sklearn

## Steps followed for the project:-

1. First goal of the project was to gather data and to cater to this step we used packages like **beautifulsoup,selenium and we also used Webloader from langchain** to gather meaningful data from website. The webiste we used for this step was cazton.com .

2. Once we scrapped the data the next step was to format the data in a format that is necessary for fine tuning an open source llm.

    2.1 Data Cleaning - After scrapping data with diff packages the next thing we had to do was to clean the data.So we first of all removed all duplicate **questions** from list of total 2200+ question answer pairs generated.Then **eyeballed** left over questions because this is the most crucial part if we are Instruction Tuning a LLM with input output pairs. So removed question answer pairs which were unnecessary and made corrections to question answer to which I could so that we have good quality instruction set at the end.

3. For this purpose we are using **Phi-1.5B** open source llm from **Microsoft**.

    3.1 The language model Phi-1.5 is a Transformer with 1.3 billion parameters. It was trained using the same data sources as phi-1, augmented with a new data source that consists of various NLP synthetic texts. 

    3.2 The special part for **phi-1.5B** was it was not instruction tuned  or through reinforcement learning from human feedback.

    3.3 For a safer model release, **phi** excluded generic web-crawl data sources such as common-crawl from the training. This strategy prevents direct exposure to potentially harmful online content, enhancing the model's safety without RLHF. 

4. Reason Phi-1.5B was choose:-
    4.1 The state of the art performance among models with less than 10 billion parameters.

    4.2 Hardware limitations - The macbook air with m1 because of its limited hardware capacity couldnt handle finetuning 7B models like falcon and llama2.
    
    
5. The phi-1.5B model required data in format given below :-
    5.1 The format is - question: <-question-> answer: <-answer->.
    

6. Now we have scrapped and formatted the dataset next step is to do actual heavy lifting.

7. Although we are using 1.5B LLm but still it takes alot of time if we just directly tune all 1.5 billion parametrs of LLM. So for faster fine tuning we are using a technqiue called as LORA (Low rank adaption).
    7.1 What is LORA?
    Low-rank adaptation refers to a technique used in machine learning, particularly in matrix factorization models. It involves updating the parameters of a model by modifying a low-rank approximation of the original parameter matrix. This
    process helps adapt the model to new data efficiently while maintaining computational tractability. Low-rank adaptation is commonly employed in collaborative filtering and recommendation systems, allowing the model to incorporate new information
    without recalculating the entire parameter matrix.
    
    
8. Inference time : After doing Fine tuning the final task was to see the performance of model on unseen data (i.e Inference).

    8.1 The biggest issue that comes with phi-1.5B even after fine-tuning is that at times it gives some not so required text as a part of the answer to the question. To solve that issue what we can do is use **Openai -> gpt-3.5-turbo-instruct** and
    whatever answer we get from phi-1.5B before giving it to end user with the help of GPT-3.5 clean it and then it give further.
    


### Files

1. training_scrapper.py - This script uses selenium to scrape data from cazton.com/     trainings and once we have got the data using gpt-3.5-turbo-instruct we have generated appropriate question-answer pair for the text.
    
2. consulting_scrapper.py - This script use beautifulsoup and gpt-3.5. So it uses bs4to   extract data from cazton.com/consulting/<domain-name> and once we have got the data using gpt-3.5-turbo-instruct we have generated appropriate **question-answer** pair for the text.

3. scrapping_webloader/selenium_scrapping.ipynb - This script uses langchain webloader to scrape data and use gpt-3.5 to make 
    menanigful questions out of data. Why it was used ? Because it reduces the effort require to write custom bs4 and selenium
    scripts to pull data from each webpage of cazton.com .
    
4. All the other csv files are intermediatery and have been made while during the process. The final file is
    **complete_data.csv** that has all the instruction pairs required for finetuning.
    
5. tester-dataset_creation_phi.ipynb - This file is used just to handle intermediatery file operatiosn and some data formatting
    that was required. 

## Inference and Fientuning:
1.  For public inference i have uploaded the fine-tuned model on https://huggingface.co/spaces/Sahibsingh12/cazton-phi
2. The notebook for inference can be find here - https://www.kaggle.com/sahib12/inference-phi-1-5

3. For Finteuning the code is present here - https://www.kaggle.com/sahib12/finetune-phi-1-5

4. The Finetuned model is here  - https://huggingface.co/Sahibsingh12/phi-1-5-finetuned-cazton_complete

    
## Refrences :-

1. https://lightning.ai/pages/community/tutorial/lora-llm/
2. https://www.youtube.com/watch?v=eC6Hd1hFvos&t=1184s
3. https://www.youtube.com/watch?v=dA-NhCtrrVE
4. https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/finetune_lora.md


## Video
<video src="https://youtu.be/Xf_pHr-oFyY" controls="controls" style="max-width: 730px;">
</video>


# Benchmarking against Zephyr

