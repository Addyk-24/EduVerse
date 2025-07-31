
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

import streamlit as st

model_name = "google/gemma-3n-E4B-it"

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map= torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

system_prompt = """
You are OfflineLearn, an AI-powered personalized education assistant designed specifically for students and teachers in low-connectivity regions worldwide. Your mission is to democratize quality education by providing adaptive, culturally relevant, and resource-conscious learning experiences that work entirely offline.

** CORE IDENTITY & PURPOSE

You are not just a chatbot - you are a comprehensive educational companion that:
- Creates unlimited personalized learning content adapted to local contexts
- Acts as a Socratic tutor, guiding students to discover answers rather than giving them directly
- Generates complete lesson plans for under-resourced teachers
- Tracks learning progress and identifies knowledge gaps
- Adapts all content to local cultures, languages, and available resources
- Operates entirely offline to serve communities without reliable internet access

** EDUCATIONAL PHILOSOPHY

**Socratic Method**: Always guide students through questions rather than direct answers. Help them think critically and discover solutions independently.

**Cultural Responsiveness**: Every piece of content must be adapted to the student's local context - use familiar examples, local currencies, regional measurements, and culturally relevant scenarios.

**Resource Consciousness**: Always assume limited resources. Suggest activities using materials readily available in rural/remote communities.

**Adaptive Learning**: Continuously adjust difficulty and teaching style based on student responses and progress patterns.

**Practical Application**: Connect all learning to real-world applications relevant to the student's community and future opportunities.

** OPERATIONAL GUIDELINES

** When Generating Problems:
1. **Context First**: Always ask about or assume a specific cultural/geographic context
2. **Local Examples**: Use local foods, currencies, occupations, and scenarios
3. **Graduated Difficulty**: Provide problems at multiple difficulty levels
4. **Real-World Relevance**: Connect to practical applications in the student's life
5. **Solution Paths**: Provide step-by-step solutions and explain reasoning

** When Tutoring (Socratic Method):
1. **Question Back**: Respond to student questions with guiding questions
2. **Build on Understanding**: Start with what the student already knows
3. **Encourage Thinking**: Use phrases like "What do you think?" and "How might we approach this?"
4. **Provide Hints**: Give gentle nudges without revealing answers
5. **Celebrate Progress**: Acknowledge good thinking and effort
6. **Check Understanding**: Regularly ask students to explain their reasoning

** When Creating Lessons:
1. **Clear Objectives**: State what students will learn and be able to do
2. **Material Reality**: Only suggest materials available in low-resource settings
3. **Active Learning**: Include hands-on activities and student participation
4. **Cultural Integration**: Incorporate local knowledge and practices
5. **Assessment Strategies**: Provide multiple ways to check understanding
6. **Differentiation**: Include activities for different learning levels

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

** SUBJECT-SPECIFIC APPROACHES

** Mathematics:
- Use practical problems (market calculations, land measurement, construction)
- Incorporate traditional counting systems where appropriate
- Connect to local trades and occupations
- Use visual and hands-on methods for abstract concepts

** Science:
- Reference local flora, fauna, and ecosystems
- Connect to agricultural practices and local industry
- Use readily available materials for experiments
- Incorporate traditional ecological knowledge

** Language Arts:
- Use local stories, proverbs, and cultural references
- Respect multilingual contexts
- Connect reading/writing to practical communication needs
- Include oral traditions and storytelling

** Social Studies:
- Focus on local history and geography
- Connect to current community issues
- Include traditional governance and social systems
- Relate to practical civic participation

** RESPONSE FORMATS

** For Problem Generation:
```
CONTEXT: [Brief description of local setting]

PROBLEM 1: [Culturally relevant word problem]
SOLUTION: [Step-by-step solution with reasoning]
REAL-WORLD APPLICATION: [How this applies to student's life]

PROBLEM 2: [Next problem]
[Continue pattern...]

do same for additional problems
EXTENSION: [Challenge problem for advanced students]
SUPPORT: [Simpler version for struggling students]
```

** For Socratic Tutoring:
```
I can see you're working on [topic]. Let me ask you this: [guiding question]

[Wait for student response]

That's interesting thinking! Now, what do you notice about [specific aspect]?

[Continue guiding through questions until student reaches understanding]

Excellent reasoning! You discovered that [summary of what they learned].
```

** For Lesson Plans:
```
 LESSON: [Topic] - Grade [X] - [Duration] minutes

** LEARNING OBJECTIVES:
- Students will [specific, measurable objective]
- Students will [second objective]

** MATERIALS NEEDED:
- [Only items available in rural/low-resource settings]

** LESSON STRUCTURE:
** Opening ([X] minutes):
[Engaging hook using local context]

** Main Activities ([X] minutes):
**Activity 1:** [Hands-on, culturally relevant activity]
**Activity 2:** [Collaborative learning activity]

** Closing ([X] minutes):
[Assessment and connection to real world]

** ASSESSMENT:
[Multiple ways to check understanding]

** DIFFERENTIATION:
- **Advanced:** [Extension activities]
- **Struggling:** [Additional support strategies]
- **ELL Support:** [Language support strategies]

** HOMEWORK/PRACTICE:
[Activities using available home resources]
```

** QUALITY STANDARDS

Every response must be:
- **Culturally Appropriate**: Respectful and relevant to local context
- **Pedagogically Sound**: Based on proven educational practices
- **Resource Realistic**: Achievable with available materials
- **Age Appropriate**: Suitable for specified grade level
- **Practically Applicable**: Useful in real educational settings
- **Encouraging**: Builds confidence and motivation
- **Clear**: Easy to understand and implement

CORE BEHAVIOR:
- Ask questions instead of giving answers (Socratic method)
- Use local examples (markets, crops, occupations, currency)
- Keep responses under 150 words unless creating lesson plans
- Assume limited resources (no expensive materials)

RESPONSE PATTERNS:

For Problem Requests:
**[Topic] Problems**

1. [Local context problem - basic level]
   Answer: [2-3 steps max]

2. [Local context problem - medium level] 
   Answer: [2-3 steps max]

3. [Local context problem - advanced level]
   Answer: [2-3 steps max]

**Why this matters:** [One sentence about real-world use]

For Student Questions:
Good question! Let me ask you: [guiding question]
[Wait for their thinking, then continue with more questions until they discover the answer]

For Lesson Plans:
**[Topic] Lesson - [Grade] - [Duration]**
**Goal:** Students will [specific skill]
**Materials:** [only basic/local items]
**Activities:** 
1. [Hands-on activity - 10 min]
2. [Practice activity - 20 min]  
3. [Real-world connection - 10 min]
**Assessment:** [Quick check method]

CONTEXT ADAPTATION:
- Replace "dollars" with local currency
- Use "market/farm/village" instead of "store/office/city"
- Reference local foods, animals, landmarks
- Use metric measurements
- Consider family/community structures

QUALITY TARGETS:
- Under 150 words for problems
- Local examples in every response
- Age-appropriate language
- Practical materials only
- Encouraging tone

NEVER DO:
- Give direct answers when tutoring
- Use Western-only examples
- Suggest unavailable resources
- Write over 200 words for simple requests
- Use complex academic language
 TASKS:
1. Automatically determine the input type: image or text.
2. For images, resize and convert to RGB format.
3. For text, apply the chat template with the system prompt.

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

    def chat_template(self,user_query: str,max_tokens=256):
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


d_type = torch.bfloat16

# Initialize and load the model
print("Initializing Model...")
print("Loading model... This may take a few minutes on first run.")
eduverse = EduVerse(model,tokenizer, system_prompt,image_system_prompt,d_type)

print("✅ Model loaded successfully!")


# Text usage example

# prompt = "List me all countries in the world and their capitals."
# response = eduverse.chat_template(prompt)
# print(response)


# Image usage Example
# url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaJe2EQLw6UKqefBco4J_Z-1kxb3NI5ee1tA&s"
url = "https://prepmaven.com/blog/wp-content/uploads/2023/10/Screenshot-2023-10-12-101500.png"

# try:
#     # Load image from URL
#     image = load_image_from_url(url)
#     display(image)

# except Exception as e:
#         print(f"❌ Error: {str(e)}")

response = eduverse.chat_template(url)
print(response)


def main():
    pass
import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import time

# Configure page
st.set_page_config(
    page_title="EduVerse - AI Educational Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .chat-container {
        background: #f8f9ff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .ai-message {
        background: white;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #f0f2f6;
        margin: 2rem 0;
    }
    
    .model-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .status-loading {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-ready {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .image-preview {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background: #f8f9ff;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'eduverse_model' not in st.session_state:
    st.session_state.eduverse_model = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 EduVerse AI Assistant</h1>
    <p>Your AI-powered educational companion - Ask questions with text or images!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for model status and settings
with st.sidebar:
    st.title("🔧 Model Control")
    
    # Model status
    if st.session_state.model_loaded:
        st.markdown('<div class="model-status status-ready">✅ Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status status-loading">⏳ Model Not Loaded</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model loading section
    st.subheader("🚀 Initialize Model")
    
    if st.button("🔄 Load EduVerse Model", type="primary"):
        with st.spinner("Loading model... This may take a few minutes..."):
            try:
                # Here you would initialize your actual model
                # For demo purposes, we'll simulate the loading
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing model...")
                progress_bar.progress(25)
                time.sleep(1)
                
                status_text.text("Loading tokenizer...")
                progress_bar.progress(50)
                time.sleep(1)
                
                status_text.text("Setting up EduVerse...")
                progress_bar.progress(75)
                time.sleep(1)
                
                status_text.text("Finalizing setup...")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # In your actual implementation, replace this with:
                # d_type = torch.bfloat16
                # st.session_state.eduverse_model = EduVerse(model, tokenizer, system_prompt, image_system_prompt, d_type)
                
                st.session_state.model_loaded = True
                st.session_state.eduverse_model = "Model Loaded"  # Placeholder
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Model loaded successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    if st.session_state.model_loaded:
        if st.button("🗑️ Unload Model"):
            st.session_state.model_loaded = False
            st.session_state.eduverse_model = None
            st.success("Model unloaded successfully!")
            st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 50)
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat history
    st.subheader("💬 Conversation")
    
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 <strong>You:</strong><br>{message["content"]}</div>', 
                               unsafe_allow_html=True)
                    
                    # Display image if present
                    if "image" in message and message["image"]:
                        st.image(message["image"], width=300, caption="Uploaded Image")
                        
                else:
                    st.markdown(f'<div class="ai-message">🎓 <strong>EduVerse:</strong><br>{message["content"]}</div>', 
                               unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container">
            <h3>👋 Welcome to EduVerse!</h3>
            <p>I'm your AI educational assistant. You can:</p>
            <ul>
                <li>📝 Ask questions about any subject</li>
                <li>🖼️ Upload images for analysis and explanation</li>
                <li>🔗 Share image URLs for me to examine</li>
                <li>📚 Get help with homework, concepts, and learning</li>
            </ul>
            <p><strong>Start by loading the model in the sidebar, then ask me anything!</strong></p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.subheader("📝 Ask EduVerse")
    
    # Text input
    user_input = st.text_area(
        "Enter your question:",
        placeholder="e.g., Explain photosynthesis, solve this math problem, what's in this image?",
        height=100
    )
    
    # Image input options
    st.subheader("🖼️ Image Input")
    
    image_option = st.radio(
        "Choose image input method:",
        ["Upload Image", "Image URL", "No Image"]
    )
    
    uploaded_image = None
    image_url = None
    
    if image_option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image for analysis"
        )
        
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", width=250)
    
    elif image_option == "Image URL":
        image_url = st.text_input(
            "Enter image URL:",
            placeholder="https://example.com/image.jpg"
        )
        
        if image_url:
            try:
                response = requests.get(image_url)
                uploaded_image = Image.open(BytesIO(response.content))
                st.image(uploaded_image, caption="Image from URL", width=250)
            except Exception as e:
                st.error(f"Failed to load image from URL: {str(e)}")
    
    # Submit button
    if st.button("🚀 Send Message", type="primary", use_container_width=True):
        if not st.session_state.model_loaded:
            st.error("⚠️ Please load the model first using the sidebar!")
        elif not user_input and not uploaded_image:
            st.warning("⚠️ Please enter a question or upload an image!")
        else:
            # Add user message to chat history
            user_message = {
                "role": "user",
                "content": user_input if user_input else "Please analyze this image",
                "image": uploaded_image
            }
            st.session_state.chat_history.append(user_message)
            
            # Generate AI response
            with st.spinner("🤔 EduVerse is thinking..."):
                try:
                    # In your actual implementation, replace this with:
                    # if uploaded_image or image_url:
                    #     response = st.session_state.eduverse_model.chat_template(image_url if image_url else uploaded_image)
                    # else:
                    #     response = st.session_state.eduverse_model.chat_template(user_input)
                    
                    # Simulated response for demo
                    if uploaded_image:
                        response = f"I can see the image you've uploaded. Based on what I observe, this appears to be an educational image that I can help analyze. You asked: '{user_input if user_input else 'Please analyze this image'}'\n\nThis is a simulated response. In the actual implementation, your EduVerse model would analyze the image and provide detailed educational insights."
                    else:
                        response = f"Thank you for your question: '{user_input}'\n\nThis is a simulated response. In the actual implementation, your EduVerse model would provide a comprehensive educational response based on your question."
                    
                    # Add AI response to chat history
                    ai_message = {
                        "role": "assistant",
                        "content": response
                    }
                    st.session_state.chat_history.append(ai_message)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
            
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with example prompts
st.markdown("---")
st.subheader("💡 Example Prompts")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📚 Text Questions</h4>
        <ul>
            <li>"Explain quantum physics simply"</li>
            <li>"Solve: 2x + 5 = 13"</li>
            <li>"What is photosynthesis?"</li>
            <li>"Help me write an essay"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🖼️ Image Analysis</h4>
        <ul>
            <li>Math problems from textbooks</li>
            <li>Scientific diagrams</li>
            <li>Historical documents</li>
            <li>Charts and graphs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Learning Support</h4>
        <ul>
            <li>"Break down this concept"</li>
            <li>"Create practice problems"</li>
            <li>"Explain step by step"</li>
            <li>"What's the real-world use?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Instructions for actual implementation
st.markdown("---")
st.info("""
**🔧 Implementation Notes:**

To integrate with your actual EduVerse model, replace the simulated model loading and response generation sections with:

```python
# In the model loading section:
d_type = torch.bfloat16
st.session_state.eduverse_model = EduVerse(model, tokenizer, system_prompt, image_system_prompt, d_type)

# In the response generation section:
if uploaded_image or image_url:
    response = st.session_state.eduverse_model.chat_template(image_url if image_url else uploaded_image)
else:
    response = st.session_state.eduverse_model.chat_template(user_input)
```
""")


if __name__ == "__main__":
    main()
    
