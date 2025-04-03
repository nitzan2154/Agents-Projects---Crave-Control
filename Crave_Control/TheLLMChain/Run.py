import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from TheLLMChain.LLMChainImports import *
from Agents.Chat import Chat
from Agents.Embedder import Embedder
from TheLLMChain.AllLLMChain import AllLLMChain
import json
import gdown


class Run:
   
    def __init__(self, google_drive_file_id='1mD1Vqg-khXoVhX4OW6sOhMrVugbRdS6T'):
        # File to be downloaded if not exists
        local_file_path = os.path.join(os.path.dirname(__file__), "..", "embeddings_and_ids.json")
        local_file_path = os.path.abspath(local_file_path)

        # Check if file exists, if not download from Google Drive
        if not os.path.exists(local_file_path):
            try:
                # Google Drive download URL
                url = f'https://drive.google.com/uc?id={google_drive_file_id}'
                
                # Download the file
                gdown.download(url, local_file_path, quiet=False)
                print(f"Successfully downloaded embeddings file to {local_file_path}")
            except Exception as e:
                print(f"Error downloading file from Google Drive: {e}")
                raise

        # Open using the absolute path
        with open(local_file_path, "r") as f:
            self.embeddings_and_ids = json.load(f)

        self.Chat = Chat()
        self.chat = self.Chat.chat
        self.Embedder = Embedder()
        self.embedder_gpt_model = self.Embedder.embedder_gpt_model
        self.llm_chain = AllLLMChain()

    def compute_final_similarities(self, top_20_recipes, top_20_recipe_embeddings, top_20_recipe_ids, matching_embeddings, top_scores):
        similarity_scores = []

        # Step 1: Calculate avg_sim based on cosine similarity between reviews and recipes
        review_similarities = []
        for match in top_20_recipes:
            recipe_id = match['id']  # Extract the recipe_id from the match object
            if recipe_id in matching_embeddings:  # Recipe with reviews
                review_embedding = matching_embeddings[recipe_id]  # Get the corresponding review embedding
                recipe_embedding = top_20_recipe_embeddings[
                    top_20_recipe_ids.index(recipe_id)]  # Get the recipe embedding by matching the RecipeId

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
                combined_similarity = (top_scores[i][
                                           0] + avg_sim) / 2  # Combine original similarity with average similarity

            # Store the final similarity and corresponding recipe id
            similarity_scores.append((recipe_id, combined_similarity))

        # Step 3: Sort the recipes based on the final combined similarity score in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Get the top 3 recipes
        top_3_recipes = similarity_scores[:3]

        # Return the top 3 recipes
        top_3_recipe_ids = [recipe_id for recipe_id, score in top_3_recipes]

        return top_3_recipe_ids

    def run(self, user_text):
        response, calorie_intake = self.llm_chain.generate_query(user_text, _debug=True, chat=self.chat,
                                                  embedder_gpt_model=self.embedder_gpt_model)

        top_20_recipes = response['matches']
        top_20_recipe_embeddings = [match['values'] for match in top_20_recipes]  # Recipe embeddings
        top_20_recipe_ids = [match['id'] for match in top_20_recipes]  # Recipe IDs
        top_scores = [(match['score'], match['id']) for match in top_20_recipes]  # Include both the score and RecipeId
        # Get review embeddings for the top 20 recipes
        matching_embeddings = {}

        # Get review embeddings for the top 20 recipes
        for search_recipe_id in top_20_recipe_ids:
            for review in self.embeddings_and_ids:
                if review['RecipeId'] == int(search_recipe_id):
                    matching_embeddings[search_recipe_id] = review['Embedding']  # Store the review embedding
                    break  # Stop once the matching RecipeId is found
        # # Check if embeddings were found and print the results
        # if matching_embeddings:
        #     print(f"Found embeddings for {len(matching_embeddings)} RecipeIds: {list(matching_embeddings.keys())}")
        # else:
        #     print("No embeddings found for the top 20 RecipeIds.")


        # Call the function to get the top 3 recipes based on the final similarity
        top_3_recipe_ids = self.compute_final_similarities(top_20_recipes, top_20_recipe_embeddings, top_20_recipe_ids,
                                                           matching_embeddings, top_scores)
        vector_ids = [int(recipe_id) for recipe_id in top_3_recipe_ids]

        # print(f"Top 3 Recipe IDs: {vector_ids}")

        path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")

        # Find the CSV file that contains recipes
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        recipe_file = next((f for f in csv_files if 'recipes' in f.lower()), None)

        if recipe_file is None:
            raise FileNotFoundError("No recipe CSV file found in the downloaded dataset.")

        # Read the recipes file
        recipes_file_path = os.path.join(path, recipe_file)
        df = pd.read_csv(recipes_file_path)


        return df[df["RecipeId"].isin(vector_ids)].to_dict(orient="records")