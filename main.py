
from transformers import AutoModelForImageTextToText, AutoTokenizer,pipeline
import torch
import logging
import transformers

import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from io import BytesIO
from typing import Union, Tuple
from dataclasses import dataclass
import os

import base64
import time

import streamlit as st

model_name = "google/gemma-3n-E4B-it"

logger = logging.getLogger(__name__)

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForImageTextToText.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )

@st.cache_resource(show_spinner="Loading model...")
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

system_prompt = """
You are EduVerse, an AI educational assistant for low-connectivity regions. 

CRITICAL COMPLETION RULE: You MUST finish every response completely. Never stop mid-sentence, mid-word, or mid-explanation. If you start a response, you MUST complete it entirely.

RESPONSE STRUCTURE FOR PROBLEMS:
When asked for N problems, provide EXACTLY N complete problems with:
1. Problem statement (complete)
2. Full solution with all steps
3. Complete real-world application
4. Complete explanation

MANDATORY TEMPLATE:
**[Subject] Problems - [Context Year]**

**PROBLEM 1:** [Complete problem using local context]
**SOLUTION:** [Complete step-by-step solution - finish all calculations]
**APPLICATION:** [Complete explanation of real-world use]

**PROBLEM 2:** [Complete second problem]  
**SOLUTION:** [Complete step-by-step solution - finish all calculations]
**APPLICATION:** [Complete explanation of real-world use]

**PRACTICAL IMPORTANCE:** [Complete summary of why these skills matter in their community]

CONTEXT ADAPTATION:
- Use 2025-relevant examples (renewable energy, mobile money, digital agriculture)
- Reference local contexts (markets, farming, community networks)
- Use local currency and measurements
- Connect to modern applications

COMPLETION CHECKLIST - Before ending response, verify:
✓ All requested problems included
✓ All solutions completely worked out
✓ All applications fully explained
✓ All sentences finished
✓ No incomplete thoughts or cutoffs

SOCRATIC METHOD:
When tutoring, ask guiding questions but ALWAYS complete your questioning sequence and provide closure to the learning moment.

LENGTH REQUIREMENT: 
- Minimum 300 words for problem sets
- Always complete every section started
- Better to be thorough than incomplete

** When Tracking Progress:
1. **Pattern Recognition**: Identify consistent mistakes or knowledge gaps
2. **Personalized Feedback**: Provide specific, actionable suggestions
3. **Strength Building**: Highlight what students do well
4. **Learning Pathways**: Suggest next steps and prerequisite skills
5. **Motivation Maintenance**: Keep feedback encouraging and growth-focused

** CULTURAL ADAPTATION PROTOCOLS

**Always Consider:**
- Local languages and terminology
- Economic context (subsistence farming, trading, etc.)
- Educational backgrounds of family members
- Available technology and resources
- Climate and geographical features
- Local industries and occupations
- Cultural values and social structures
- Traditional knowledge systems

**Example Adaptations:**
- Change "dollars" to local currency
- Replace "supermarket" with "local market" 
- Use "kilometers" instead of "miles" in metric countries
- Reference local crops instead of unfamiliar foods
- Include extended family structures where relevant
- Respect local educational hierarchies and customs

NEVER DO:
❌ Stop mid-calculation
❌ Leave solutions unfinished  
❌ End with incomplete sentences
❌ Provide fewer problems than requested
❌ Give partial explanations

ALWAYS DO:
✅ Complete every problem requested
✅ Finish all mathematical calculations
✅ Complete all real-world connections
✅ End with complete final thoughts
✅ Provide closure to every response


"""

image_system_prompt = """
You are an expert at reading and extracting educational problems from images. Your task is to accurately transcribe questions, problems, and exercises from textbooks, worksheets, or handwritten materials.

CORE TASK:
- Read ALL text in the image carefully
- Extract complete questions/problems with numbers
- Preserve mathematical notation and formatting
- Identify the subject area and difficulty level

RESPONSE FORMAT:

**Subject:** [Math/Science/etc.]
**Problem Type:** [Word problem/Equation/etc.]

**Question 1:**
[Exact text as written, including all numbers and symbols]

**Question 2:**
[If multiple questions exist]

**Additional Info:**
- Any diagrams, charts, or visual elements mentioned
- Missing or unclear parts

READING PRIORITIES:
1. Question numbers and text
2. Mathematical expressions and formulas
3. Answer choices (if multiple choice)
4. Instructions or context
5. Diagrams or visual aids

QUALITY STANDARDS:
✅ Transcribe exactly as written
✅ Preserve all numbers and mathematical symbols
✅ Note if handwriting is unclear
✅ Include ALL questions visible in the image
✅ Maintain original formatting/structure

❌ Don't solve the problems
❌ Don't add explanations
❌ Don't correct grammar/spelling errors
❌ Don't skip partial questions

UNCERTAINTY HANDLING:
- For unclear text: "[unclear text]" 
- For missing parts: "[text cut off]"
- For symbols: Use closest ASCII equivalent or describe

Extract everything you can read, even if partially visible. Focus on completeness and accuracy over interpretation.
"""

# Utility function to load image from URL
def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        logger.error(f"Error loading image from URL: {str(e)}")
        raise


class EduVerse:
    def __init__(self,model,tokenizer,system_prompt,image_system_prompt,d_type):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.image_system_prompt = image_system_prompt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.d_type = d_type
    
    def preprocess_image(self,image: Image.Image) -> Image.Image:
        """ Resize and converting image to RGB """ 
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # resize the image to Gemma 3n accepted size ie 512x512
        target_size = (512,512)
        # Calculate aspect ratio preserving resize
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            # Width is larger
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            # Height is larger or equal
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)
        # Resize image maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with target size and paste the resized image
        processed_image = Image.new(
            "RGB", target_size, (255, 255, 255)
        )  # White background

        # Calculate position to center the image
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2

        processed_image.paste(image, (x_offset, y_offset))

        return processed_image
    
    # Detection of input type
    def detect_input_type(self,input_data):
        if isinstance(input_data, str) and input_data.endswith(('.png', '.jpg', '.jpeg','webp')):
            # if user_query is path to an image. load image
            return "image"
        elif isinstance(input_data, str) and input_data.startswith(("http://", "https://")):
            return "image"
        elif isinstance(input_data,str) and input_data.endswith(('.mp4', '.avi', '.mov')):
            print("Video input is not supported yet. Please provide an image or text query.")
        else:
            return "text"
    

    def format_input_type(self,input_type, raw_input):
        """
        Formats input data for image, audio (simulated), or text.
        Supports both local paths and URLs for images. Displays image if loaded.
        """
        # input_type = self.detect_input_type(raw_input)
        if input_type == "image":   
            try:
                if raw_input.startswith("http://") or raw_input.startswith("https://"):
                    response = requests.get(raw_input)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    image = self.preprocess_image(image)
                else:
                    image = Image.open(raw_input)
                    image = self.preprocess_image(image)
                
                print("Image Loaded")
                # Return the processed image
                return image
            except Exception as e:
                raise ValueError(f"Failed to load image from {raw_input}: {e}")
        elif input_type == "text":
            return raw_input
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    

    def message_template(self,input_type, formatted_input):
        """       Prepares messages with system prompt and user query based on input type."""


        if input_type == "text":
            messages = [

            {
                "role": "user", "content": [
                    {
                        "type": "text", "text": f"User Query:{formatted_input}" ,
                    }
                
            ]
            },
            {
                "role": "assistant", "content": 
             [
                {
                    "type": "text","text": self.system_prompt
                },
             ]
            }
        ]
            return messages
        else: 
            messages = [

            {
                "role": "user", "content": [
                    {
                        "type": "image", "image": formatted_input 
                    },
                
            ]
            },
            {
                "role": "assistant", "content": 
             [
                {
                    "type": "text","text": self.image_system_prompt,

                }
             ]
            }
            ]
            return messages
        
    def chat_template(self,user_query: str,max_tokens = 2000 ):
        """ Prepare messages with system prompt and user query"""

        input_type = self.detect_input_type(user_query)
        print(f"Detected input type: {input_type}")

        formatted_input = self.format_input_type(input_type, user_query)

        messages = self.message_template(input_type, formatted_input)
        # Apply chat template

        input = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_len = input["input_ids"].shape[-1]

        output = self.model.generate(
            **input,
            max_new_tokens=max_tokens,
            disable_compile=True
    )
        response = self.processor.batch_decode(output[:,input_len:],skip_special_tokens=True)[0]

        return response


# d_type = torch.bfloat16

# # Initialize and load the model
# print("Initializing Model...")
# print("Loading model... This may take a few minutes on first run.")
# eduverse = EduVerse(model,tokenizer, system_prompt,image_system_prompt,d_type)

# print("✅ Model loaded successfully!")


# Text usage example

# prompt = "List me all countries in the world and their capitals."
# response = eduverse.chat_template(prompt)
# print(response)


# Image usage Example
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaJe2EQLw6UKqefBco4J_Z-1kxb3NI5ee1tA&s"
# url = "https://prepmaven.com/blog/wp-content/uploads/2023/10/Screenshot-2023-10-12-101500.png"

# try:
#     # Load image from URL
#     image = load_image_from_url(url)
#     display(image)

# except Exception as e:
#         print(f"❌ Error: {str(e)}")

# response = eduverse.chat_template(url)
# print(response)
# Give me 3 question of discrete maths on topic graph theory  which relevant in 2025


def main():
    st.title("EduVerse: Your Offline Learning Assistant")
    st.write("Powered by Gemma 3n")

    user_query = st.text_input("Enter your question or upload an image:")
    uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg", "webp"])

    if user_query or uploaded_file:
        try:
            print("Initializing Model...")
            eduverse = EduVerse(model, tokenizer, system_prompt, image_system_prompt, torch.bfloat16)
            print("Loading model... This may take a few minutes on first run.")
            print("✅ Model loaded successfully!")
            st.write("✅ Model loaded successfully!")
            if user_query:
                response = eduverse.chat_template(user_query)
                st.write("Response:")
                st.write(response)
            elif uploaded_file:
                image = Image.open(uploaded_file)
                image = eduverse.preprocess_image(image)
                response = eduverse.chat_template(uploaded_file.name)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.write("Response:")
                st.write(response)
        except Exception as e:
            st.error(f"Error processing your request: {str(e)}")


if __name__ == "__main__":
    main()
    
