import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.AgentImports import *

load_dotenv("env")

CHAT_DEPLOYMENT = "team1-gpt4o"
CHAT_API_VERSION = "2023-05-15"
EMBEDDINGS_API_VERSION = "2024-08-01-preview"

class Embedder:

    def __init__(self):
        # Load needed variables
        self.AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
        self.EMBEDDING_DEPLOYMENT = "team1-embedding"
        self.EMBEDDING_MODEL_NAME = "text-embedding-3-small"
        self.AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
        self.EMBEDDING_API_VERSION = "2024-08-01-preview"

        # Initialize the Azure OpenAI chat model
        self.embedder_gpt_model = AzureOpenAIEmbeddings(
            model = self.EMBEDDING_MODEL_NAME,
            azure_endpoint = self.AZURE_OPENAI_ENDPOINT,
            azure_deployment = self.EMBEDDING_DEPLOYMENT,
            api_key = self.AZURE_OPENAI_API_KEY,
            api_version = self.EMBEDDING_API_VERSION,
            openai_api_type = "azure",
        )


