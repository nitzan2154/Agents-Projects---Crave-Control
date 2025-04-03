import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Agents.AgentImports import *
from TheLLMChain.LLMChainImports import *
from User.UserImports import *
from TheLLMChain.Run import Run
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from Agents.feedback import FeedbackAgent
from Display_All import display_recipes_ranked

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

class Crave_Control_Demo:

    def __init__(self):
        self.system_prompt = """
        Hi there! I'm CraveControl, your friendly meal-finding assistant.
    
        I'm here to help you discover the perfect meal that matches your ideal calorie needs and taste preferences.
        
        To get started, could you share a bit about yourself and your current cravings?
        
        Let me know details like your age, weight, height, gender, workout habits, any dietary preferences or restrictions,
        and even what ingredients you have on hand. Your input will help me tailor the best, healthiest meal options just for you!
        
        Input: 
        """
        self.user_input = input(self.system_prompt)
        self.Run = Run()
        self.feedback_agent = FeedbackAgent()
        self.process()

    def process(self):
    # First run
        records = self.Run.run(self.user_input)
        display_recipes_ranked(records)

        # Start feedback loop
        current_input = self.user_input
        while True:
            feedback_prompt = input("\nWould you like to give feedback on the recipe? (yes/no): ").strip().lower()
            if feedback_prompt != "yes":
                print("Thank you for using CraveControl! ")
                break

            # Get feedback
            feedback = self.feedback_agent.collect_feedback()

            # Modify the structured input JSON
            modified_structured_input = self.feedback_agent.modify_structured_input(current_input, feedback)

            # Re-run recommendation with modified input
            records = self.Run.run(modified_structured_input)
            display_recipes_ranked(records)

            # Update current input for potential further loops
            current_input = modified_structured_input

if __name__ == "__main__":
    crave_agent = Crave_Control_Demo()


