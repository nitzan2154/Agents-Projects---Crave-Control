import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from User.UserImports import *
from dotenv import load_dotenv
from User.UserInput import UserInput
from Agents.Chat import Chat

load_dotenv()

class Parser:
    def __init__(self):
        # Initialize the output parser with the updated schema
        self.parser = PydanticOutputParser(pydantic_object=UserInput)

        self.Chat = Chat()
        self.chat = self.Chat.chat

    def calculate_bmi(self, weight_kg, height_cm):
        """Calculate BMI using the formula: weight (kg) / height (m)^2"""
        height_m = height_cm / 100  # Convert cm to meters
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)

    def classify_bmi(self, bmi):
        """Classify BMI into categories"""
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal weight"
        elif 25 <= bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    def calculate_tdee(self, weight_kg, height_cm, age, gender, activity_level):
        """Calculate TDEE (Total Daily Energy Expenditure) using Mifflin-St Jeor Equation"""
        if gender.lower() == "male":
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

        # Activity level mapping
        activity_multipliers = {
            1: 1.2,  # Sedentary
            2: 1.375,  # Light activity
            3: 1.55,  # Moderate activity
            4: 1.725,  # Active
            5: 1.9  # Very active
        }

        # Get multiplier (default to sedentary if out of range)
        multiplier = activity_multipliers.get(activity_level, 1.2)

        # Calculate TDEE
        return int(bmr * multiplier)

    def adjust_calories_based_on_bmi(self, calories, bmi_class):
        """Modify calorie intake based on BMI classification"""
        bmi_modifiers = {
            "Underweight": 1.15,  # Increase by 15%
            "Normal weight": 1.0,  # No change
            "Overweight": 0.9,  # Reduce by 10%
            "Obese": 0.8  # Reduce by 20%
        }
        return int(calories * bmi_modifiers.get(bmi_class, 1.0))  # Default to no change

    def distribute_calories_per_meal(self, total_calories, meal_type):
        """Distribute total daily calories based on meal type"""
        meal_distribution = {
            "breakfast": 0.3,  # 30% of daily intake
            "lunch": 0.4,  # 40%
            "dinner": 0.25,  # 25%
            "snack": 0.15  # 15%
        }
        return int(total_calories * meal_distribution.get(meal_type.lower(), 0.3))  # Default to 30%

    def generate_struct(self, user_text, _debug=False):
        prompt_template = """
        Extract and structure the following user text into JSON format according to the provided schema.

        Follow this **exact JSON schema**:
        {format_instructions}

        Only output JSON, without explanations or extra text.

        User Text: {user_text}
        """

        instructions = self.parser.get_format_instructions()

        prompt = PromptTemplate(
            input_variables=["user_text"],
            partial_variables={"format_instructions": instructions},
            template=prompt_template
        )

        formatted_prompt = prompt.format(user_text=user_text)

        # Call the LLM
        llm_output = self.chat.invoke(formatted_prompt)

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
            bmi = self.calculate_bmi(weight, height)
            bmi_class = self.classify_bmi(bmi)
            tdee = self.calculate_tdee(weight, height, age, gender, activity_level)
            adjusted_calories = self.adjust_calories_based_on_bmi(tdee, bmi_class)
            meal_calories = self.distribute_calories_per_meal(adjusted_calories, meal_type)

            # Add calculated calories to output
            json_output["calorie_intake"] = meal_calories
            # json_output["response_format"] = "response_format"  # Placeholder for response_format without quotes

            return json_output

        except json.JSONDecodeError:
            raise Exception("LLM output is not valid JSON.")
        except Exception as e:
            raise Exception(f"Parsing error: {str(e)}")


# if __name__ == "__main__":

    # Parser = Parser()
    #
    # # Example User Input
    # user_profile = {
    #     "age": 25,
    #     "weight": 77,
    #     "height": 160,
    #     "calorie_limit": 600,
    #     "gender": "female",
    #     "activity_level": 3,
    #     "meal_type": "breakfast"
    # }
    #
    # # Step 1: Calculate BMI
    # bmi = Parser.calculate_bmi(user_profile["weight"], user_profile["height"])
    # print(bmi)
    # bmi_class = Parser.classify_bmi(bmi)
    #
    # # Step 2: Calculate Daily Calories (TDEE)
    # tdee = Parser.calculate_tdee(
    #     user_profile["weight"],
    #     user_profile["height"],
    #     user_profile["age"],
    #     user_profile["gender"],
    #     user_profile["activity_level"]
    # )
    #
    # # Step 3: Modify Calories Based on BMI Class
    # adjusted_calories = Parser.adjust_calories_based_on_bmi(tdee, bmi_class)
    #
    # # Step 4: Get Meal-Specific Calories
    # meal_calories = Parser.distribute_calories_per_meal(adjusted_calories, user_profile["meal_type"])
    #
    # # Display Results
    # print(f" User BMI: {bmi} ({bmi_class})")
    # print(f" Daily Calorie Needs (TDEE): {tdee} kcal")
    # print(f" Adjusted Calories (After BMI Modifier): {adjusted_calories} kcal")
    # print(f" Recommended Calories for {user_profile['meal_type'].capitalize()}: {meal_calories} kcal")
