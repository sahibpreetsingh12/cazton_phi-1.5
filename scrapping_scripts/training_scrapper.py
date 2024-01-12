from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import os
import openai
import shutup

shutup.please()
os.environ["OPENAI_API_KEY"] = '---open-ai-key--'

openai_key = os.getenv('OPENAI_API_KEY')
# Path to your Chrome WebDriver
webdriver_path = '/opt/homebrew/bin/chromedriver'

# URL of the website
url = 'https://cazton.com/trainings'

# Set up Chrome options (optional)
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')  # Run Chrome in headless mode (without opening a window)

# Create a Chrome webdriver
service = Service(webdriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Navigate to the URL
driver.get(url)

# Find the element(s) and extract text
training_elements = driver.find_elements(By.CLASS_NAME, 'small.text-center span')  


answer_list =[]
question_list =[]

# initialize the models
openai = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    openai_api_key=openai_key
)
for i,element in enumerate(training_elements):

    text_of_p_tag = element.text
    # print(f" ---> {i}",element.text,"\n\n") 


    template = """Frame the question based on the context below. If the
        question cannot be framed using the information provided answer
        with "I don't know".

        Context: Let's think of the situation. You are given text now after reading text your task is 
        to think of ideal question for that text. These texts are  about different trainings that are 
        offered by cazton. Your questions should be about what things are offered in trainings mentioned 
        in the {text}

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
        print("\n\n",response , ' ---> ',text_of_p_tag,"\n\n")

        answer_list.append(text_of_p_tag)
        question_list.append(response)
 

# Close the browser
driver.quit()

# dictionary of lists 
dict = {'questions': question_list, 'answers': answer_list} 

df_temp = pd.DataFrame(dict)
    
df = pd.read_csv('question_answer.csv')



df = pd.concat([df,df_temp],ignore_index=True)



print(len(df))

df.to_csv('question_answer.csv')