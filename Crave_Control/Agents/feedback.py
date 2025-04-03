import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.AgentImports import *
from typing import List
from User.UserImports import *
from typing import Dict, List
from User.Parser import Parser

class FeedbackAgent:
    def __init__(self):
        self.parser = Parser()

    def get_feedback_prompt(self, recipe_name: str) -> str:
        return f"How was the recipe '{recipe_name}'? Do you want to change any ingredients?"

    def collect_feedback(self) -> Dict:
        print("\n Feedback Time!")
        add = input("Any ingredients you'd like to ADD? (comma-separated, or leave blank): ").strip()
        remove = input("Any ingredients you'd like to REMOVE? (comma-separated, or leave blank): ").strip()

        return {
            "satisfied": False,
            "add_ingredients": [i.strip() for i in add.split(",") if i.strip()],
            "remove_ingredients": [i.strip() for i in remove.split(",") if i.strip()]
        }
    
    def modify_structured_input(self, original_text: str, feedback: Dict) -> Dict:
        structured = self.parser.generate_struct(original_text)

        original_ingredients = set(structured.get("included_ingredients", []))
        to_add = set(feedback.get("add_ingredients", []))
        to_remove = set(feedback.get("remove_ingredients", []))

        updated_ingredients = list((original_ingredients | to_add) - to_remove)
        structured["included_ingredients"] = updated_ingredients

        # print(original_ingredients)
        # print(updated_ingredients)

        return structured