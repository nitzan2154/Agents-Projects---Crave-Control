import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.AgentImports import *

load_dotenv("env")

class Pine:

    def __init__(self):
        # Load needed variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

        # Initialize the Pinecone object and its index
        self.pc = Pinecone(
                            api_key=self.PINECONE_API_KEY
                        )

        self.index = self.pc.Index("crave-agent")


