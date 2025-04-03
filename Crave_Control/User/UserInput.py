import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from User.UserImports import *

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

