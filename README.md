# 🎓 EduVerse: "Your Personal Learning Universe, Anywhere"

> *Democratizing quality education through offline-first, culturally adaptive AI tutoring*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Gemma](https://img.shields.io/badge/Powered%20by-Gemma%203-green)](https://ai.google.dev/gemma)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## 🌍 Problem Statement

**263 million children worldwide lack access to quality education.** Many live in low-connectivity regions where traditional online educational tools fail. Teachers often lack resources and training, while students struggle with content that doesn't reflect their cultural context or practical needs.

EduVerse bridges this gap by providing an **AI-powered educational assistant that works entirely offline**, adapts to local cultures, and requires minimal resources.

## ✨ What Makes EduVerse Different

### 🔄 **Adaptive Learning Engine**
- Dynamically adjusts difficulty based on student performance
- Identifies knowledge gaps and creates personalized study paths
- Learns from student interaction patterns to improve over time

### 🌏 **Cultural Context Awareness**
- Automatically adapts content to local contexts (farming, markets, local currencies)
- Uses familiar examples and culturally relevant scenarios
- Respects local educational traditions and knowledge systems

### 🎯 **Socratic Teaching Method**
- Guides students through discovery rather than giving direct answers
- Builds critical thinking skills through strategic questioning
- Encourages student confidence and independent problem-solving

### 📶 **Offline-First Design**
- Works completely without internet connection
- No data collection or privacy concerns
- Runs on modest hardware (laptop, tablet, or basic computer)

### 🎨 **Resource-Conscious Teaching**
- Suggests activities using locally available materials
- Creates lesson plans for under-resourced classrooms
- Provides alternatives for expensive educational tools

## 🚀 Key Features

### For Students
- **📚 Unlimited Practice Problems**: Contextually relevant exercises that never repeat
- **🤔 Socratic Tutoring**: AI guide that helps you think through problems
- **📈 Progress Tracking**: Personalized feedback and learning path recommendations
- **🌟 Multi-Subject Support**: Math, Science, Language Arts, and more
- **🏠 Homework Help**: Step-by-step guidance without giving away answers

### For Teachers
- **📋 Lesson Plan Generator**: Complete lesson plans adapted to your resources
- **📊 Student Assessment**: Tools to track class progress and identify struggling students
- **🎓 Professional Development**: Teaching strategy suggestions and classroom management tips
- **📝 Content Creation**: Generate worksheets, quizzes, and activities
- **🌐 Curriculum Alignment**: Content aligned with local educational standards

### For Communities
- **👨‍👩‍👧‍👦 Family Engagement**: Helps parents support their children's education
- **📱 Mobile-Friendly**: Works on smartphones and tablets
- **🔋 Low Power Usage**: Designed for areas with limited electricity
- **💰 Cost-Effective**: No subscription fees or ongoing costs

## 🛠️ Technology Stack

- **AI Model**: Gemma 3n (27B parameters)
- **Backend**: Python with transformers

## 📦 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 50GB available storage

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/EduVerse.git
cd EduVerse
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the application:**
```bash
streamlit run app.py
```

7. **Access the interface:**
Open your browser to `http://localhost:8501`

## 🎯 Usage Examples

### For Students

**Getting Help with Math:**
```
Student: "I don't understand how to solve 2x + 5 = 13"

EduVerse: "I can see you're working with an equation! Let me ask you this: 
What do you think our goal is when we have an equation like this? What are we 
trying to find?"

[Guides student through Socratic questioning to discover the solution]
```

**Practice Problems:**
```
Request: "Generate 3 fraction problems for grade 6 students in rural Kenya"

EduVerse: Creates problems about:
- Dividing farmland between family members
- Calculating portions of maize harvest
- Sharing water containers in the community
```

### For Teachers

**Lesson Plan Generation:**
```
Input: "Create a 45-minute science lesson about photosynthesis for grade 7, 
limited resources, rural Tanzania"

Output: Complete lesson plan with:
- Local plant examples and materials
- Hands-on activities using available items
- Assessment strategies
- Cultural connections to farming
```

**Student Assessment:**
```
Upload student work → Receive detailed analysis:
- Strengths and weaknesses identified
- Personalized study recommendations
- Progress tracking over time
- Parent communication suggestions
```

## 🌟 Challenges faced

1. 🌐 Lack of Internet Dependency

    Challenge: Running large language models (LLMs) like Gemma 3n in an offline environment.

    Solution: Used quantized versions, optimized runtimes (GGUF, ONNX), and model distillation to fit within resource-constrained devices.

2. 💾 Limited Hardware Resources
   
    Challenge: Most rural areas only have access to low-end devices with limited CPU/GPU and RAM.

    Solution: static compilation to reduce memory footprint.

4. 🧠 Real-Time Adaptivity Without Cloud

   Challenge: Adapting lessons in real time typically relies on server-backed AI models.

    Solution: Embedded mini-engines using distilled transformers that adapt locally based on cached learning progress.

5. 🗣️ Contextual Content Generation

   Challenge: Generating examples or problems relevant to a student’s region, language, or curriculum without internet access.

    Solution: Built a local knowledge base with region-specific datasets (e.g., rural economy, agriculture), and fine-tuned prompts for localized responses.

6. 📊 Images tensor being saved has null or nan value

   Challenge: When processing the image, the tensor were have value of null or nan 

    Solution: Used sampling, clamp and normalize logits and Replace NaN/inf values

7. 🔐 Model Fragmentation Across Devices

    Challenge: Memory fragmentation and multi-GPU inconsistencies with mixed-precision models.

    Solution: Used accelerate with device_map="balanced_low_0" and offloaded parts to CPU where needed. Also handled CUDA errors and fallback logic.

8. 🧪 Evaluation Without Benchmarks

   Challenge: No standard datasets exist for offline personalized education systems.

    Solution: Created custom test cases, offline simulation environments, and used teacher feedback for evaluation.


## 🎨 User Interface

<img width="754" height="502" alt="UI" src="https://github.com/user-attachments/assets/6eb0cb7e-22c7-4403-98b5-1df2876c4282" />

### Student 
- **📚 Subject Selection**: Easy navigation between subjects
- **🎯 Practice Mode**: Unlimited problems with instant feedback
- **🤝 Tutor Chat**: Conversational learning with AI guide

### Teacher 
- **📋 Lesson Planning**: Generate and customize lesson plans
- **📊 Analytics**: Class performance insights and recommendations
- **📝 Content Creation**: Create custom exercises and assessments


## 🌐 Cultural Adaptation

EduVerse automatically adapts content for different regions:

### Supported Contexts
- **🌾 Rural farming communities** (Kenya, Tanzania, India, etc.)
- **🏪 Trading/market towns** (West Africa)

## YOUTUBE VIDEO

[YOUTUBE VIDEO](https://youtu.be/E99L5BXhrmA)

