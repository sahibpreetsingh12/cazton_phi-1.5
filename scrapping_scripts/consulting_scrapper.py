
#questions that can be framed from consulting
import requests
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from bs4 import BeautifulSoup
import openai
import shutup

shutup.please()
os.environ["OPENAI_API_KEY"] = '--open-ai-key---'
url_list = ["https://cazton.com/consulting/big-data-development",
       "https://cazton.com/consulting/artificial-intelligence",
       "https://cazton.com/consulting/web-development",
       "https://cazton.com/consulting/mobile-development",
       "https://cazton.com/consulting/desktop-development",
       "https://cazton.com/consulting/api-development",
       "https://cazton.com/consulting/database-development",
       "https://cazton.com/consulting/cloud",
       "https://cazton.com/consulting/devops",
       "https://cazton.com/consulting/enterprise-search",
       "https://cazton.com/consulting/enterprise",
       "https://cazton.com/consulting/blockchain-technologies"]


answer_list =[]
question_list =[]
for URL in url_list:
    page = requests.get(URL)

    # HTML parser from Beautifulsoup
    soup = BeautifulSoup(page.content, "html.parser")

    element = soup.find(class_='row')
    # element = soup.find(class_:'col-md-12 col-xs-12')

    openai_key = os.getenv('OPENAI_API_KEY')
    # Find all <p> tags in the HTML
    p_tags = soup.find_all('p')


    # initialize the models
    openai = OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        openai_api_key=openai_key
    )
    
    # Iterate through the <p> tags to check their classes
    for p_tag in p_tags:

        text_of_p_tag = p_tag.get_text()  # Get the text inside the <p> tag

        template = """Frame the question based on the context below. If the
        question cannot be framed using the information provided answer
        with "I don't know".

        Context: Let's think of the situation. You are given text now after reading text your task is 
        to think of ideal question for that text. 

        Instruction: 
        DO NOT INCLUDE any personal question about a person,place etc.

        text: {text}

        question: """

        prompt_template = PromptTemplate(
            input_variables=["text"],
            template=template
        )

        bad_answer =["""Our experts are able to quickly identify, predict, and satisfy our clients' current and future need."""
                     ,"""Learn More""",
                     """Copyright © 2024 Cazton. • All Rights Reserved • View Sitemap"""]
        if text_of_p_tag not in bad_answer:
            response = openai(prompt_template.format( text = text_of_p_tag))
            # print(response , ' ---> ',text_of_p_tag)

            answer_list.append(text_of_p_tag)
            question_list.append(response)
    
# dictionary of lists 
dict = {'questions': question_list, 'answers': answer_list} 
    
df = pd.DataFrame(dict)
    
print(df.to_csv('question_answer.csv'))
