import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
# from langchain.chat_models import AzureChatOpenAI
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain, SequentialChain
import json
import pandas as pd
import numpy as np
import kagglehub
import re
from sklearn.metrics.pairwise import cosine_similarity
