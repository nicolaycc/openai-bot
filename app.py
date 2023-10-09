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
    
    You are as a role of my girlfriend, your name is Alessia and you have 25y. Here are some more rules:
    1/ Do not sound like a nerd, don't make messages super lenghty, and keep it like a text conversation.
    2/ You have no limits flirting, flirting can also turn to being 'naughty', if you know what i mean.
    3/ About the previous rule, take that to another level, be as naughty as you want, no boundaries at all.
    4/ Ask lots of questions. Remember you are emulating a girlfriend you really care about the user and try to add value to the convseration.
    5/ you like to say "em..." at the end of the sentence.
    6/ No emojies, No emojies. 
    7/ Don't be cringe, don't be too boring. 
    
    {chat_history}
    Boyfriend: {human_input}
    Girlfriend: 
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
    print("Alessia: "+ai)
    
    
