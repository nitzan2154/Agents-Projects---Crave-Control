from pydantic import BaseModel, Field
from typing import List
import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI
