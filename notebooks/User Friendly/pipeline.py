from dotenv import load_dotenv
from pprint import pprint
import getpass
import os
from openai import OpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from openai import OpenAI, AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone

load_dotenv("env")

AZURE_OPENAI_API_KEY =  os.getenv("API_KEY")
EMBEDDINGS_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

api_key = os.getenv("API_KEY") # Changed from 'AZURE_OPENAI_API_KEY' to 'API_KEY'
if not api_key:
    raise Exception(" API_KEY is missing! Please set it in your environment.") # Changed from 'AZURE_OPENAI_API_KEY' to 'API_KEY'
else:
    print("API Key is set and ready to use.")
    

AZURE_OPENAI_API_KEY = os.getenv('API_KEY')
EMBEDDINGS_DEPLOYMENT = "team1-embedding"
EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"
AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
CHAT_DEPLOYMENT = "team1-gpt4o"
CHAT_API_VERSION = "2023-05-15"
EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"
EMBEDDINGS_API_VERSION = "2024-08-01-preview"


client = AzureOpenAI(
    api_key = AZURE_OPENAI_API_KEY, # api key
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_version = "2024-08-01-preview",
    azure_deployment = EMBEDDINGS_DEPLOYMENT,
)

embedder_gpt_model = AzureOpenAIEmbeddings(
            model = EMBEDDINGS_MODEL_NAME,
            azure_endpoint = AZURE_OPENAI_ENDPOINT,
            azure_deployment = EMBEDDINGS_DEPLOYMENT,
            api_key = AZURE_OPENAI_API_KEY,
            api_version = EMBEDDINGS_API_VERSION,
            openai_api_type = "azure",
        )

chat = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=CHAT_API_VERSION,
    openai_api_type="azure",
    temperature=0.7
)
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index("crave-agent")

from pydantic import BaseModel, Field
from typing import List
import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List


class UserInput(BaseModel):
    age: int = Field(description="User's age in years")
    weight: float = Field(description="User's weight in kg")
    height: float = Field(description="User's height in cm")
    gender: str = Field(description="User's gender (male/female)")
    activity_level: int = Field(description="User's activity level (1: sedentary, 2: light, 3: moderate, 4: active, 5: very active)")
    meal_type: str = Field(description="Type of meal user wants to eat (breakfast, lunch, dinner, snack)")
    food_category: str = Field(description="Category of the food (e.g., Mexican, Italian, etc.)")
    cravings: str = Field(description="The type of craving (e.g., sweet, salty, etc.)")
    included_ingredients: List[str] = Field(description="Ingredients available to cook")
    calorie_intake: int = Field(description="Calculated calorie intake for this meal based on user's data")
    #response_format: str = Field(description="The format of the response")

# Initialize the output parser with the updated schema
parser = PydanticOutputParser(pydantic_object=UserInput)


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI using the formula: weight (kg) / height (m)^2"""
    height_m = height_cm / 100  # Convert cm to meters
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

def classify_bmi(bmi):
    """Classify BMI into categories"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"
    
def calculate_tdee(weight_kg, height_cm, age, gender, activity_level):
    """Calculate TDEE (Total Daily Energy Expenditure) using Mifflin-St Jeor Equation"""
    if gender.lower() == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    # Activity level mapping
    activity_multipliers = {
        1: 1.2,    # Sedentary
        2: 1.375,  # Light activity
        3: 1.55,   # Moderate activity
        4: 1.725,  # Active
        5: 1.9     # Very active
    }

    # Get multiplier (default to sedentary if out of range)
    multiplier = activity_multipliers.get(activity_level, 1.2)

    # Calculate TDEE
    return int(bmr * multiplier)

def adjust_calories_based_on_bmi(calories, bmi_class):
    """Modify calorie intake based on BMI classification"""
    bmi_modifiers = {
        "Underweight": 1.15,  # Increase by 15%
        "Normal weight": 1.0,  # No change
        "Overweight": 0.9,  # Reduce by 10%
        "Obese": 0.8  # Reduce by 20%
    }
    return int(calories * bmi_modifiers.get(bmi_class, 1.0))  # Default to no change

def distribute_calories_per_meal(total_calories, meal_type):
    """Distribute total daily calories based on meal type"""
    meal_distribution = {
        "breakfast": 0.3,  # 30% of daily intake
        "lunch": 0.4,      # 40%
        "dinner": 0.25,    # 25%
        "snack": 0.15      # 15%
    }
    return int(total_calories * meal_distribution.get(meal_type.lower(), 0.3))  # Default to 30%

def generate_struct(user_text, _debug=False):
    prompt_template = """
    Extract and structure the following user text into JSON format according to the provided schema.

    Follow this **exact JSON schema**:
    {format_instructions}

    Only output JSON, without explanations or extra text.

    User Text: {user_text}
    """

    instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["user_text"],
        partial_variables={"format_instructions": instructions},
        template=prompt_template
    )

    formatted_prompt = prompt.format(user_text=user_text)

    # Call the LLM
    llm_output = chat.invoke(formatted_prompt)

    # Extract raw text from LLM output
    json_string = llm_output.content  # Extract JSON text

    # Remove unwanted Markdown formatting (```json ... ```)
    json_string = json_string.strip("```json").strip("```").strip()

    try:
        # Convert cleaned JSON string to Python dict
        json_output = json.loads(json_string)

        # Extract user profile details for calorie calculations
        profile = json_output
        weight = profile["weight"]
        height = profile["height"]
        age = profile["age"]
        gender = profile["gender"]
        activity_level = profile["activity_level"]
        meal_type = profile["meal_type"]
        food_category = profile["food_category"]
        cravings = profile["cravings"]
        included_ingredients = profile["included_ingredients"]

        # Calculate calories
        bmi = calculate_bmi(weight, height)
        bmi_class = classify_bmi(bmi)
        tdee = calculate_tdee(weight, height, age, gender, activity_level)
        adjusted_calories = adjust_calories_based_on_bmi(tdee, bmi_class)
        meal_calories = distribute_calories_per_meal(adjusted_calories, meal_type)

        # Add calculated calories to output
        json_output["calorie_intake"] = meal_calories
        #json_output["response_format"] = "response_format"  # Placeholder for response_format without quotes

        return json_output

    except json.JSONDecodeError:
        raise Exception("LLM output is not valid JSON.")
    except Exception as e:
        raise Exception(f"Parsing error: {str(e)}")
    
from openai import OpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
import os
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import AzureChatOpenAI

def build_query_chain(llm):

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
            'age', 'weight', 'gender', 'activity_level', 'meal_type', 'cravings', 'calorie_intake', 'response_format'
            ],
        output_variables=['query'],
        verbose=False
    )

    return chain, response_format, output_parser

import json

# Function to generate query and perform similarity search
def generate_query(user_text, _debug=True, chat=None, embedder_gpt_model=None):
    # Generate structured output from the user text
    parsed_output = generate_struct(user_text, _debug=_debug) # Calorie intake is calcualted here

    # Use the chain and response format to generate a query
    chain, response_format, output_parser = build_query_chain(chat)
    parsed_output['response_format']=response_format

    # Running the chain to generate the query
    query_output = chain.run(**parsed_output)
    query = query_output[27:-7]  # Extract the query part
    calorie_intake = parsed_output['calorie_intake']

    # Define calorie range based on input query
    calorie_range_max = int(parsed_output['calorie_intake'] * 1.1)  # 10% higher
    calorie_range_min = int(parsed_output['calorie_intake'] * 0.6)  # 60% lower

    # Convert query into embeddings
    query_embedding = embedder_gpt_model.embed_query(query)  # Function to convert the query into vector representation

    # Perform the similarity search in Pinecone
    response = index.query(
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

with open("embeddings_and_ids.json", "r") as f:
    embeddings_and_ids = json.load(f)


def run(user_text):
    response, calorie_intake = generate_query(user_text, _debug=True, chat=chat, embedder_gpt_model=embedder_gpt_model)

    top_20_recipes = response['matches']
    top_20_recipe_embeddings = [match['values'] for match in top_20_recipes]  # Recipe embeddings
    top_20_recipe_ids = [match['id'] for match in top_20_recipes]  # Recipe IDs
    top_scores = [(match['score'], match['id']) for match in top_20_recipes]  # Include both the score and RecipeId
    # Get review embeddings for the top 20 recipes
    matching_embeddings = {}

    # Get review embeddings for the top 20 recipes
    for search_recipe_id in top_20_recipe_ids:
        for review in embeddings_and_ids:
            if review['RecipeId'] == int(search_recipe_id):
                matching_embeddings[search_recipe_id] = review['Embedding']  # Store the review embedding
                break  # Stop once the matching RecipeId is found
    # Check if embeddings were found and print the results
    if matching_embeddings:
        print(f"Found embeddings for {len(matching_embeddings)} RecipeIds: {list(matching_embeddings.keys())}")
    else:
        print("No embeddings found for the top 20 RecipeIds.")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    def compute_final_similarities(top_20_recipes, top_20_recipe_embeddings, matching_embeddings, top_scores):
        similarity_scores = []

        # Step 1: Calculate avg_sim based on cosine similarity between reviews and recipes
        review_similarities = []
        for match in top_20_recipes:
            recipe_id = match['id']  # Extract the recipe_id from the match object
            if recipe_id in matching_embeddings:  # Recipe with reviews
                review_embedding = matching_embeddings[recipe_id]  # Get the corresponding review embedding
                recipe_embedding = top_20_recipe_embeddings[top_20_recipe_ids.index(recipe_id)]  # Get the recipe embedding by matching the RecipeId

                if recipe_embedding is not None:
                    # Calculate cosine similarity between the recipe and the review
                    cosine_sim = cosine_similarity([recipe_embedding], [review_embedding])[0][0]
                    review_similarities.append(cosine_sim)

        # Calculate the average cosine similarity for the reviews
        avg_sim = np.mean(review_similarities) if review_similarities else 0  # If no reviews, set avg_sim to 0

        # Step 2: Iterate through the recipes and compute the final similarity
        for i, match in enumerate(top_20_recipes):
            recipe_id = match['id']
            if recipe_id in matching_embeddings:  # Recipe with reviews
                review_embedding = matching_embeddings[recipe_id]  # Get the corresponding review embedding
                recipe_embedding = top_20_recipe_embeddings[i]  # Get the recipe embedding

                if recipe_embedding is not None:
                    # Calculate cosine similarity between the recipe and the review
                    cosine_sim = cosine_similarity([recipe_embedding], [review_embedding])[0][0]

                    # Compute the average similarity (original similarity + review similarity)
                    combined_similarity = (top_scores[i][0] + cosine_sim) / 2
            else:  # Recipe without reviews
                combined_similarity = (top_scores[i][0] + avg_sim) / 2  # Combine original similarity with average similarity

            # Store the final similarity and corresponding recipe id
            similarity_scores.append((recipe_id, combined_similarity))

        # Step 3: Sort the recipes based on the final combined similarity score in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Get the top 3 recipes
        top_3_recipes = similarity_scores[:3]

        # Return the top 3 recipes
        top_3_recipe_ids = [recipe_id for recipe_id, score in top_3_recipes]

        return top_3_recipe_ids


    # Call the function to get the top 3 recipes based on the final similarity
    top_3_recipe_ids = compute_final_similarities(top_20_recipes, top_20_recipe_embeddings, matching_embeddings, top_scores)
    vector_ids = [int(recipe_id) for recipe_id in top_3_recipe_ids]

    print(f"Top 3 Recipe IDs: {vector_ids}")

    import pandas as pd
    import numpy as np
    import kagglehub
    import re
    import os
    # Download latest version
    path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    recipes_file_path = os.path.join(path, csv_files[1])
    df = pd.read_csv(recipes_file_path)

    return df[df["RecipeId"].isin(vector_ids)].to_dict(orient="records")