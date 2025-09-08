from langchain_openai import OpenAI
from dotenv import load_dotenv  # TO Load API keys from .env file
load_dotenv() # load kya openai key ko jo platform openai se milega or .env file me store karenge
llm = OpenAI(temperature=0.9) # temperature is used to control the randomness of the output
print(llm("What is the capital of pakistan?")) # prompt