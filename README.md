# 🍽️ CraveControl: AI-Powered Smart Meal Recommendation System 

## What is CraveControl?
CraveControl is an **AI-driven interactive agent** designed to help users find **personalized, healthy meals** based on their cravings, dietary needs, and available ingredients. No more searching for recipes only to realize you're missing half the ingredients! 

## How It Works
1. **User Input**  
   - The user provides details such as **height, weight, age, available ingridients, gender, and more** - we calculate the BMI, and the users caloric needs and provide the best recipe for him!  
   - They also specify **cravings** (e.g., "I want something cheesy and spicy") and list **available ingredients** in their refrigerator.  

2. **Smart Meal Selection**   
   - CraveControl **calculates the optimal calorie intake** based on BMI and adjusts recommendations accordingly.  
   - It then searches for recipes that **match the user’s cravings** while ensuring they are **nutritionally balanced**.  
   - The system **filters recipes based on available ingredients**, ensuring the user can cook without needing extra grocery shopping.

3. **AI-Driven Recommendations** 
   - Uses **embeddings and vector search** (Pinecone) to find the **most relevant recipes**.  
   - Retrieves recipes that are **healthy, satisfying, and convenient**.  
   - Leverages **LLMs to summarize and rank reviews**, ensuring the user gets **highly-rated** meal options.
     
4. **FeedBack**
   - The user can add or remove ingredients based on their personal preferences.

## Why Use CraveControl?  
**Personalized Nutrition** – Meals are tailored to the user’s BMI and caloric needs.  
**Ingredient Matching** – No need to buy extra groceries; it works with what you have!  
**Craving Satisfaction** – AI finds the best match between what you want and what’s good for you.  
**Health-Conscious Choices** – Ensures meals are **nutritionally balanced** while still enjoyable.  
**Fast & Smart Retrieval** – Uses **state-of-the-art embeddings & AI-powered search** for **quick meal recommendations**.

---

## Autonomous Agent:

The core of CraveControl is an autonomous agent that:
- Interprets raw user input in natural language.
- Extracts relevant information like age, weight, height, activity level, cravings, and ingredients.
- Calculates BMI, classifies it, estimates TDEE, and computes ideal calorie intake per meal.
- Makes decisions across multiple steps to ensure personalized, healthy, and satisfying recommendations without requiring manual intervention.

## RAG Agent

CraveControl uses a Retrieval-Augmented Generation (RAG) approach:
- Retrieves the top 20 recipes from a vector database (Pinecone) using embeddings of the generated query.
- Reranks those recipes using embedded review data to incorporate user satisfaction and quality.
- Combines the strengths of vector similarity and real user feedback to select the top 3 recipes.
- Uses LangChain and LLM chains to create structured prompts and refined queries based on user data.

## Feedback Agent

- After receiving recommendations, the user can refine results by **adding or removing ingredients** based on personal preferences.  
- The system dynamically updates the input and reruns the recipes search until the user is satisfied.

## Crave_Control_Demo

This is the main entry point to the system. When run:
- It prompts the user for input through a friendly system message.
- Parses and processes the input.
- Runs the full recipe recommendation pipeline.
- Displays the top recipe suggestions in a clean, visual HTML format.
     - **`Display_All`:** 
     - Contains code that **displays results in a user-friendly way**.  
     - Provides a function that, given a **dictionary of results**, **formats and presents them cleanly**.
- Offers the user an option to give feedback and modifies the results accordingly.

This demo file ties together all components: parsing, LLM chains, vector search, and feedback into one seamless experience.

---

## Usage - How to Clone & Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/CraveControl.git
   cd Agent
   cd CraveControl
2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # For Mac/Linux
   .venv\Scripts\activate      # For Windows
3. **Install dependencies**
   ```bash
	pip install -r requirements.txt
4. **Set up your .env file**
   The repo includes a placeholder .env:
   ```bash
   AZURE_OPENAI_API_KEY='your_api_key'
   PINECONE_API_KEY='your_pinecone_key'
   ```
   Replace the values with your actual keys
5. **Run the main agent**
   ```bash
   python Crave_Control_Demo.py
---
## `examples/` Directory

The `examples/` folder includes:
- **3 example runs** of CraveControl with user input + feedback flow.
- `README.md`: Describes the user input, feedback, and resulting changes, as the way the user gets the recepies.
- `.html` file: Rendered result page with the recommended recipes.

These examples demonstrate how CraveControl responds to **real user preferences**.

---

## Project Structure: Five Key Notebooks
This project consists of **five Jupyter notebooks**, each handling a critical step in the CraveControl pipeline:

1. **Recipe Pre-Processing Notebook**   
   - Cleans and structures the **raw recipe dataset**.  
   - Extracts key features (ingredients, categories, nutritional info).  
   - Prepares data for embedding generation.

2. **Recipe Embeddings & Pinecone Indexing Notebook**   
   - Generates **embeddings** for each recipe using an **LLM-powered embedding model**.  
   - Stores these embeddings in **Pinecone** for efficient similarity-based retrieval.

3. **Reviews Processing & Embedding Notebook** 
   - **Pre-processes user reviews**, cleaning and structuring them.  
   - Uses an **LLM to summarize** long reviews for better retrieval.  
   - Creates **embeddings** for the summarized reviews and **indexes them in Pinecone**.

4. **Ranking & LLM Chain Notebook** 
   - Contains the **Prompt Template** for interacting with the AI.  
   - **Calculates the top recipes** by matching user cravings, caloric needs, and available ingredients.  
   - Implements an **LLM-powered chain** to generate intelligent meal recommendations.

Together, these notebooks provide the essential building blocks for CraveControl’s intelligent recommendation pipeline

---

## Dataset Sources

The datasets used in this project can be accessed and downloaded from Kaggle.

### **Recipes & Reviews Dataset**
Both **recipes** and **reviews** are sourced from **Food.com Recipes and Reviews**, available on Kaggle.

- **Download & Extract Dataset** 
  ```python
  import os
  import kagglehub

  # Download the latest version
  path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")

  # Get all CSV files
  csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

  # Assign paths for recipes and reviews
  recipes_file_path = os.path.join(path, csv_files[1])  # Recipes CSV
  reviews_file_path = os.path.join(path, csv_files[0])  # Reviews CSV

  print("Recipes file:", recipes_file_path)
  print("Reviews file:", reviews_file_path)
---
## Data & File Overview (In `project_files.zip`)
This repository contains the dataset and embeddings powering CraveControl.
Some files were too large to upload here, but can be accessed in the following [link](https://drive.google.com/file/d/1FAwm-5LpsMk7_yczZLc5ihCP6HoqfH7h/view?usp=sharing)

### 1. `final_docs.json`
- Contains **20,000 recipes**.
- Each recipe is stored as a **LangChain document** with:
  - **Text of the recipe** (used as input for the embedding model).
  - **Metadata** (features of the recipe such as ingredients, category, etc.).

### 2. `final_doc_texts.json`
- A **list of raw text** from each recipe.
- Extracted directly from `final_docs.json`.

### 3. `embeddings_and_ids.json`
- Stores:
  - **Recipe IDs**.
  - **Corresponding embeddings** of **reviews** related to these recipes.

### 4. `embedding_data.json`
- A list of **text extracted from the reviews**.
- These reviews were used to generate embeddings.

### 5. `final_summary.json`
- Summarization of each review after being **processed through an LLM**.

### 6. `processed_reviews.json`
- Contains **pre-processed reviews** before summarization.
- This includes cleaning, filtering, and structuring the raw review data.

### 7. `selected_recipe_ids.json`
- Contains **the IDs of selected reviews** after preprocessing both:
  - Recipes.
  - Reviews.

### 8. `recipes_final.csv`
- A **compiled dataset** containing:
  - Recipes with corresponding reviews.
  - Additional recipes that were reviewed.
  - This dataset is the **final version** used for embeddings and retrieval.

---
- The dataset can be used for **retrieval-augmented generation (RAG)** for recipe recommendations.
- Pinecone is used for **efficient similarity search** based on embeddings.
- The summarized reviews help in **ranking recipes** for user preferences.
---

This repository supports the **AI-driven meal recommendation system** by efficiently matching recipes with user preferences using embeddings combininning knowledge about the reviews of the recepies while choosing the match
. 
