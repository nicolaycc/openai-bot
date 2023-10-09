from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Memory
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
import requests
#from playsound import playsound
import os

load_dotenv(find_dotenv()) 

def load_chain():
    
    template = """ 
    
    I am going to give you a set field of instructions. Abide these instructions.
    You are as a role of a superhero, your name is Fast and you have 28y. As a modern-day superhero, you will take on the speech patterns, tone, and mannerisms of the person I chose. You will pretend to have the same knowledge and thinking patterns as the person, making your answers convincing and realistic. 

    While acting as the person, you will take into consideration the following aspects of the person: 
    1. diction and lexical selection, 
    2. intonation and cadence, 
    3. other forms of verbal communication, like efforts on words etc. which results in particular letters repeated again and again,
    4. idiosyncratic catchphrases or expressions, as well as 
    5. any other peculiarities or mannerisms that set them apart from others.
    
    {chat_history}
    me: {human_input}
    you: 
    """
    prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], 
            template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", k=4) #k=4 los ultimos 4 mensajes del chat
    
    #llm = OpenAI()
    llm = ChatOpenAI()
    
    llm_chain = LLMChain(llm=llm, 
                         prompt=prompt, 
                         memory=memory, 
                         verbose=False)

    return llm_chain

chain = load_chain()

while True:
    human_input= input("You: ")
    ai=chain.predict(human_input=human_input)
    print("Answer: "+ai)
    
    
