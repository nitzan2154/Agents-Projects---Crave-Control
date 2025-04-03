from dotenv import load_dotenv
from pprint import pprint
import getpass
import os

# from langchain.chat_models import AzureChatOpenAI
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI

# from openai import OpenAI, AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from pinecone import Pinecone

