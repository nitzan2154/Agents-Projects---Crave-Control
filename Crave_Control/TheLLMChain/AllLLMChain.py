import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from TheLLMChain.LLMChainImports import *
from Agents.Chat import Chat
from Agents.Embedder import Embedder
from User.Parser import Parser
from Agents.Pine import Pine

class AllLLMChain:
    def __init__(self):
        self.Chat = Chat()
        self.chat = self.Chat.chat
        self.Embedder = Embedder()
        self.embedder_gpt_model = self.Embedder.embedder_gpt_model
        self.Parser = Parser()
        self.parser = self.Parser.parser
        self.Pine = Pine()
        self.index = self.Pine.index

    def build_query_chain(self, llm):
        query_string = ResponseSchema(
            name="query_string",
            description="A detailed and specific query string that reflects the user's needs, suitable for a similarity search on a vector database of healthy recipes"
        )

        output_parser = StructuredOutputParser.from_response_schemas(
            [query_string]
        )

        response_format = output_parser.get_format_instructions()

        prompt = PromptTemplate.from_template(
            template="""
            You are an expert chef, a professional recipe creator, and a certified nutritionist.
            Your task is to suggest the best recipe that fits the following user preferences,
            ensuring that the recipe is not only delicious but also healthy for the user.

            Carefully analyze and consider the following details:

            'Food Category': {food_category}  # Type of cuisine or food group (e.g., vegan, Mediterranean, low-carb)
            'Required Ingredients': {included_ingredients}  # The ingredients that must be included in the recipe (e.g., spinach, quinoa)

            **User Profile:**
            'Age': {age}  # User's age
            'Weight': {weight}  # User's weight (in kg)
            'Height': {height}  # User's height (in cm)
            'Gender': {gender}  # User's gender (male/female)
            'Activity Level': {activity_level}  # How active the user is (1: sedentary, 2: light, 3: moderate, 4: active, 5: very active)
            'Meal Type': {meal_type}  # The type of meal (breakfast, lunch, dinner, snack)
            'Cravings': {cravings}  # Any specific craving (e.g., sweet, savory, crunchy)

            As a professional nutritionist, ensure the recipe aligns with the user's health goals.
            If the user seeks to maintain or lose weight, gain muscle, or manage a specific condition,
            provide a recipe that helps meet those goals while maintaining proper nutrition and balanced calories.

            Your job is to generate a **clear and specific** query string based on these preferences to perform a similarity search
            in a vector database of recipes.

            Be sure to:
            - Focus on the key ingredients, food type, and any health requirements.
            - Suggest meals that are nutritionally balanced, considering the user's weight, age, and activity level.
            - Ensure the recipe falls within the user's required **calorie range**.
            - Use clear language that directly addresses the food needs and health considerations of the user.

            **Response format**: {response_format}
            """
        )

        query_chain = LLMChain(llm=llm, prompt=prompt, output_key='query')

        chain = SequentialChain(
            chains=[query_chain],
            input_variables=[
                'food_category', 'included_ingredients', 'height',
                'age', 'weight', 'gender', 'activity_level', 'meal_type', 'cravings', 'calorie_intake',
                'response_format'
            ],
            output_variables=['query'],
            verbose=False
        )

        return chain, response_format, output_parser

    # Function to generate query and perform similarity search
    def generate_query(self, user_text, _debug=True, chat=None, embedder_gpt_model=None):
        # Generate structured output from the user text
        parsed_output = self.Parser.generate_struct(user_text, _debug=_debug)  # Calorie intake is calcualted here

        # Use the chain and response format to generate a query
        chain, response_format, output_parser = self.build_query_chain(self.chat)
        parsed_output['response_format'] = response_format

        # Running the chain to generate the query
        query_output = chain.run(**parsed_output)
        query = query_output[27:-7]  # Extract the query part
        calorie_intake = parsed_output['calorie_intake']

        # Define calorie range based on input query
        calorie_range_max = int(parsed_output['calorie_intake'] * 1.1)  # 10% higher
        calorie_range_min = int(parsed_output['calorie_intake'] * 0.6)  # 60% lower

        # Convert query into embeddings
        query_embedding = self.embedder_gpt_model.embed_query(
            query)  # Function to convert the query into vector representation

        # Perform the similarity search in Pinecone
        response = self.index.query(
            namespace="embds+calo",  # Assuming namespace is calo
            vector=query_embedding,  # The embedding of the user input
            top_k=20,
            include_values=True,
            include_metadata=True,
            filter={
                "calories": {
                    "$lte": calorie_range_max,  # Less than or equal to max calories
                    "$gte": calorie_range_min  # Greater than or equal to min calories
                }
            }
        )

        return response, calorie_intake