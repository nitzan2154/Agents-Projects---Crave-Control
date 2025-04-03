import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import requests
from io import BytesIO
import math
import uuid
import re
import ast

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
    transformed['reviews_count'] = int(recipe_dict.get('ReviewCount', 0))
    
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
        transformed['images'] = safe_eval_list(images)
    
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
    Display multiple recipes in a compact format, ranked by relevance.
    
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
    
    # Display each recipe
    for i, recipe in enumerate(sorted_recipes):
        display_single_recipe(recipe, i+1, len(sorted_recipes))

def display_single_recipe(recipe_dict, rank=None, total=None):
    """
    Display a single recipe in a compact format.
    
    Parameters:
    recipe_dict (dict): Recipe dictionary
    rank (int, optional): Rank of this recipe
    total (int, optional): Total number of recipes
    """
    # Check if the recipe dictionary needs transformation
    if 'RecipeId' in recipe_dict:
        recipe_dict = parse_recipe_dict(recipe_dict)
    
    # Generate a unique ID for this recipe
    recipe_id = f"recipe_{uuid.uuid4().hex[:8]}"
    
    # Create HTML for recipe header with rank if provided
    rank_display = f"<span style='color: #888; font-size: 0.9em;'>Rank {rank}/{total}</span>" if rank else ""
    
    header_html = f"""
    <div style="font-family: Arial; padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="color: #333; margin: 0 0 5px 0;">{recipe_dict['Name']}</h2>
            {rank_display}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #666; font-size: 0.9em;">By {recipe_dict['Author']} | {recipe_dict['category']}</span>
            <span style="font-size: 1em; color: #f6ab00;">{'★' * int(recipe_dict['rating'])} ({recipe_dict['reviews_count']})</span>
        </div>
    </div>
    """
    display(HTML(header_html))
    
    # Display images in a horizontal scrollable container
    if recipe_dict['images'] and len(recipe_dict['images']) > 0:
        image_container = f"""
        <div style="margin: 10px 0; max-width: 100%; overflow-x: auto;">
            <div style="display: flex; gap: 5px;">
        """
        
        for img_url in recipe_dict['images']:
            image_container += f"""
            <div style="flex: 0 0 auto; width: 150px; height: 100px;">
                <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 3px;">
            </div>
            """
        
        image_container += """
            </div>
        </div>
        """
        
        display(HTML(image_container))
    
    # Create a compact info section
    info_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;'>"
    info_html += f"<div style='padding: 5px 10px; background-color: #f0f8ff; border-radius: 4px;'><b>Time:</b> {format_time(recipe_dict['time'])}</div>"
    info_html += f"<div style='padding: 5px 10px; background-color: #f0f8ff; border-radius: 4px;'><b>Calories:</b> {recipe_dict['calories']} kcal</div>"
    info_html += f"<div style='padding: 5px 10px; background-color: #f0f8ff; border-radius: 4px;'><b>Servings:</b> {recipe_dict['servings']}</div>"
    
    # Add nutrition percentages (rounded)
    info_html += f"<div style='padding: 5px 10px; background-color: #f0fff0; border-radius: 4px;'><b>Carbs:</b> {round(recipe_dict.get('carbohydrates percentage', 0), 1)}%</div>"
    info_html += f"<div style='padding: 5px 10px; background-color: #f0fff0; border-radius: 4px;'><b>Protein:</b> {round(recipe_dict.get('proteins percentage', 0), 1)}%</div>"
    info_html += f"<div style='padding: 5px 10px; background-color: #f0fff0; border-radius: 4px;'><b>Fat:</b> {round(recipe_dict.get('fat percentage', 0), 1)}%</div>"
    info_html += "</div>"
    display(HTML(info_html))
    
    # Display ingredients and instructions stacked vertically
    content_html = """
    <div style='font-family: Arial; margin: 10px 0;'>
        <h4 style='margin-top: 0; margin-bottom: 5px;'>Ingredients</h4>
        <ul style='margin: 0 0 10px 0; padding-left: 20px;'>
    """
    
    for ingredient, amount in recipe_dict.get('ingredients', {}).items():
        content_html += f"<li><b>{ingredient}:</b> {amount}</li>"
    
    content_html += """
        </ul>
        <h4 style='margin-top: 10px; margin-bottom: 5px;'>Instructions</h4>
        <p style='margin: 0;'>{}</p>
    </div>
    """.format(recipe_dict.get('instructions', 'No instructions available.'))
    
    display(HTML(content_html))
    
    # Display tags if any
    if recipe_dict.get('keywords'):
        tags_html = "<div style='margin: 5px 0;'>"
        for keyword in recipe_dict['keywords']:
            tags_html += f"<span style='background-color: #eaeaea; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; margin-right: 5px;'>{keyword}</span>"
        tags_html += "</div>"
        display(HTML(tags_html))
    
    # Add separator between recipes
    display(HTML("<hr style='margin: 20px 0; border: 0; border-top: 1px dashed #ccc;'>"))

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