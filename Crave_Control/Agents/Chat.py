import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.AgentImports import *
from dotenv import load_dotenv
import os

load_dotenv()

class Chat:

    def __init__(self):
        # Load needed variables
        self.AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
        self.CHAT_DEPLOYMENT = "team1-gpt4o"
        self.AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
        self.CHAT_API_VERSION = "2023-05-15"

        # Initialize the Azure OpenAI chat model
        self.chat = AzureChatOpenAI(
            azure_deployment=self.CHAT_DEPLOYMENT,
            api_key=self.AZURE_OPENAI_API_KEY,
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
            openai_api_version=self.CHAT_API_VERSION,
            openai_api_type="azure",
            temperature=0.7
        )



