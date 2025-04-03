import uuid
import ast
import re
import webbrowser
import os
import pandas as pd

def parse_recipe_dict(recipe_dict):
    """
    Parse and transform the recipe dictionary into the format expected by the display functions.
    
    Parameters:
    recipe_dict (dict): Raw recipe dictionary from the dataset
    
    Returns:
    dict: Transformed recipe dictionary ready for display
    """
    # Create a new dictionary with the expected structure
    transformed = {}
    
    # Basic info
    transformed['Name'] = recipe_dict.get('Name', 'Unnamed Recipe')
    transformed['Author'] = recipe_dict.get('AuthorName', 'Unknown')
    transformed['category'] = recipe_dict.get('RecipeCategory', 'Uncategorized')
    transformed['rating'] = recipe_dict.get('AggregatedRating', 0)
    review_count = recipe_dict.get('ReviewCount', 0)
    transformed['reviews_count'] = int(review_count) if pd.notna(review_count) else 0
    
    # Parse time information
    cook_time = recipe_dict.get('CookTime', '')
    prep_time = recipe_dict.get('PrepTime', '')
    total_time = recipe_dict.get('TotalTime', '')
    
    # Use total time if available, otherwise calculate from cook and prep time
    if total_time and isinstance(total_time, str):
        time_str = parse_iso_duration(total_time)
    elif prep_time and isinstance(prep_time, str):
        time_str = parse_iso_duration(prep_time)
    else:
        time_str = "00:00:00"  # Default time format
    
    transformed['time'] = time_str
    
    # Nutrition information
    transformed['calories'] = recipe_dict.get('Calories', 0)
    transformed['servings'] = recipe_dict.get('RecipeServings', 1)
    
    # Calculate nutrition percentages
    total_nutrients = (
        recipe_dict.get('CarbohydrateContent', 0) +
        recipe_dict.get('ProteinContent', 0) +
        recipe_dict.get('FatContent', 0)
    )
    
    if total_nutrients > 0:
        transformed['carbohydrates percentage'] = (recipe_dict.get('CarbohydrateContent', 0) / total_nutrients) * 100
        transformed['proteins percentage'] = (recipe_dict.get('ProteinContent', 0) / total_nutrients) * 100
        transformed['fat percentage'] = (recipe_dict.get('FatContent', 0) / total_nutrients) * 100
    else:
        transformed['carbohydrates percentage'] = 0
        transformed['proteins percentage'] = 0
        transformed['fat percentage'] = 0
    
    # Parse ingredients
    ingredients = {}
    quantities = safe_eval_list(recipe_dict.get('RecipeIngredientQuantities', '[]'))
    parts = safe_eval_list(recipe_dict.get('RecipeIngredientParts', '[]'))
    
    # Ensure quantities and parts have the same length
    min_length = min(len(quantities), len(parts))
    for i in range(min_length):
        ingredients[parts[i]] = quantities[i] if i < len(quantities) else ""
    
    transformed['ingredients'] = ingredients
    
    # Parse instructions
    instructions_list = safe_eval_list(recipe_dict.get('RecipeInstructions', '[]'))
    transformed['instructions'] = "\n".join(instructions_list)
    
    # Parse keywords
    transformed['keywords'] = safe_eval_list(recipe_dict.get('Keywords', '[]'))
    
    # Parse images
    images = recipe_dict.get('Images', 'character(0)')
    if images == 'character(0)':
        transformed['images'] = []
    else:
        # If we have actual image URLs, parse them
        # Limit to 4 and only include valid image links (simple check for "http")
        image_list = safe_eval_list(images)
        valid_images = [img for img in image_list if isinstance(img, str) and img.startswith("http")]
        transformed['images'] = valid_images[:4]
    
    return transformed

def safe_eval_list(list_str):
    """
    Safely evaluate a string representation of a list.
    
    Parameters:
    list_str (str): String representation of a list (e.g., "c('item1', 'item2')")
    
    Returns:
    list: Parsed list, or empty list if parsing fails
    """
    try:
        # Handle R-style c() lists
        if isinstance(list_str, str) and list_str.startswith('c('):
            # Extract content inside c() and parse it
            content = list_str[2:-1]
            # Split by commas, but not if the comma is inside quotes
            items = re.findall(r'"([^"]*)"|\'([^\']*)\'', content)
            # Flatten the list of tuples and filter out empty strings
            return [item for sublist in items for item in sublist if item]
        elif isinstance(list_str, list):
            return list_str
        else:
            # Try using ast.literal_eval for other list formats
            return ast.literal_eval(list_str)
    except (SyntaxError, ValueError):
        # Return empty list if parsing fails
        return []

def parse_iso_duration(iso_duration):
    """
    Parse ISO 8601 duration format (e.g., PT10M) to "HH:MM:SS" format.
    
    Parameters:
    iso_duration (str): Duration in ISO 8601 format
    
    Returns:
    str: Duration in "HH:MM:SS" format
    """
    hours, minutes, seconds = 0, 0, 0
    
    # Check if string is in ISO 8601 format
    if isinstance(iso_duration, str) and iso_duration.startswith('PT'):
        # Extract hours, minutes, seconds
        if 'H' in iso_duration:
            h_parts = iso_duration.split('H')
            hours = int(h_parts[0].replace('PT', ''))
            iso_duration = h_parts[1]
        
        if 'M' in iso_duration:
            m_parts = iso_duration.split('M')
            minutes = int(m_parts[0].replace('PT', ''))
            iso_duration = m_parts[1]
        
        if 'S' in iso_duration:
            seconds = int(iso_duration.split('S')[0].replace('PT', ''))
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def display_recipes_ranked(recipes_list, relevance_scores=None):
    """
    Generate and display recipes in a web browser.
    
    Parameters:
    recipes_list (list): List of recipe dictionaries
    relevance_scores (list, optional): List of relevance scores corresponding to recipes.
                                      If None, recipes are assumed to be pre-sorted.
    """
    # Transform all recipe dictionaries
    transformed_recipes = [parse_recipe_dict(recipe) for recipe in recipes_list]
    
    # Sort recipes by relevance if scores are provided
    if relevance_scores:
        # Pair recipes with their scores and sort
        recipe_pairs = list(zip(transformed_recipes, relevance_scores))
        recipe_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_recipes = [pair[0] for pair in recipe_pairs]
    else:
        # Assume recipes are already sorted
        sorted_recipes = transformed_recipes
    
    # Collect HTML for all recipes
    all_recipes_html = ""
    for i, recipe in enumerate(sorted_recipes):
        all_recipes_html += display_single_recipe(recipe, i+1, len(sorted_recipes))
    
    # Wrap with a full HTML page
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Recipe Collection</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
            .recipe {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
            .recipe-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
            .recipe-header h2 {{ margin: 0; color: #333; }}
            .recipe-rank {{ color: #888; font-size: 0.9em; }}
            .recipe-info {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
            .recipe-info-item {{ padding: 5px 10px; background-color: #f0f8ff; border-radius: 4px; }}
            .recipe-nutrition {{ background-color: #f0fff0; }}
            .recipe-images {{ display: flex; gap: 5px; overflow-x: auto; margin: 10px 0; }}
            .recipe-images img {{ width: 150px; height: 100px; object-fit: cover; border-radius: 3px; }}
            .recipe-ingredients ul {{ margin: 0 0 10px 0; padding-left: 20px; }}
            .recipe-keywords {{ margin: 5px 0; }}
            .recipe-keyword {{ background-color: #eaeaea; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; margin-right: 5px; }}
        </style>
    </head>
    <body>
        {all_recipes_html}
    </body>
    </html>
    """
    
    # Generate a unique filename
    output_file = f"recipes_{uuid.uuid4().hex[:8]}.html"
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    # Open the file in the default web browser
    webbrowser.open(f'file://{os.path.realpath(output_file)}')

def display_single_recipe(recipe_dict, rank=None, total=None):
    """
    Generate HTML for a single recipe.
    
    Parameters:
    recipe_dict (dict): Recipe dictionary
    rank (int, optional): Rank of this recipe
    total (int, optional): Total number of recipes
    
    Returns:
    str: HTML string for the recipe
    """
    # Check if the recipe dictionary needs transformation
    if 'RecipeId' in recipe_dict:
        recipe_dict = parse_recipe_dict(recipe_dict)
    
    # Generate a unique ID for this recipe
    recipe_id = f"recipe_{uuid.uuid4().hex[:8]}"
    
    # Rank display
    rank_display = f'<span class="recipe-rank">Rank {rank}/{total}</span>' if rank else ""
    
    # Images HTML
    images_html = ""
    if recipe_dict['images'] and len(recipe_dict['images']) > 0:
        images_html = '<div class="recipe-images">'
        for img_url in recipe_dict['images']:
            images_html += f'<img src="{img_url}" alt="Recipe Image">'
        images_html += '</div>'
    
    # Info items
    info_items = [
        f'<div class="recipe-info-item"><b>Time:</b> {format_time(recipe_dict["time"])}</div>',
        f'<div class="recipe-info-item"><b>Calories:</b> {recipe_dict["calories"]} kcal</div>',
        f'<div class="recipe-info-item"><b>Servings:</b> {recipe_dict["servings"]}</div>',
        f'<div class="recipe-info-item recipe-nutrition"><b>Carbs:</b> {round(recipe_dict.get("carbohydrates percentage", 0), 1)}%</div>',
        f'<div class="recipe-info-item recipe-nutrition"><b>Protein:</b> {round(recipe_dict.get("proteins percentage", 0), 1)}%</div>',
        f'<div class="recipe-info-item recipe-nutrition"><b>Fat:</b> {round(recipe_dict.get("fat percentage", 0), 1)}%</div>'
    ]
    
    # Ingredients HTML
    ingredients_html = '<ul>'
    for ingredient, amount in recipe_dict.get('ingredients', {}).items():
        ingredients_html += f'<li><b>{ingredient}:</b> {amount}</li>'
    ingredients_html += '</ul>'
    
    # Keywords HTML
    keywords_html = ""
    if recipe_dict.get('keywords'):
        keywords_html = '<div class="recipe-keywords">'
        for keyword in recipe_dict['keywords']:
            keywords_html += f'<span class="recipe-keyword">{keyword}</span>'
        keywords_html += '</div>'
    
    # Ratings
    rating = recipe_dict.get("rating", 0)
    reviews = recipe_dict.get("reviews_count", 0)

    if pd.notna(rating) and pd.notna(reviews):
        try:
            rating_html = f"{'★' * int(recipe_dict['rating'])} ({recipe_dict['reviews_count']})"
        except ValueError:
            rating_html = "No ratings"
    else:
        rating_html = "No ratings"
        
        
    # Construct full recipe HTML
    recipe_html = f"""
    <div id="{recipe_id}" class="recipe">
        <div class="recipe-header">
            <h2>{recipe_dict['Name']}</h2>
            {rank_display}
        </div>
        <div>
            <span>By {recipe_dict['Author']} | {recipe_dict['category']}</span>
            <span>{rating_html}</span>
        </div>
        
        {images_html}
        
        <div class="recipe-info">
            {''.join(info_items)}
        </div>
        
        <div>
            <h4>Ingredients</h4>
            {ingredients_html}
            
            <h4>Instructions</h4>
            <p>{recipe_dict.get('instructions', 'No instructions available.')}</p>
        </div>
        
        {keywords_html}
    </div>
    """
    
    return recipe_html

def format_time(time_str):
    """Format time string from "00:5:00" to "5m" format"""
    try:
        parts = time_str.split(':')
        result = []
        if int(parts[0]) > 0:
            result.append(f"{int(parts[0])}h")
        if int(parts[1]) > 0:
            result.append(f"{int(parts[1])}m")
        if not result:
            return "< 1m"
        return "".join(result)
    except:
        return time_str  # Return original if parsing fails