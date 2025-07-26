# 🎓 OfflineLearn: AI-Powered Education for Everyone

> *Democratizing quality education through offline-first, culturally adaptive AI tutoring*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Gemma](https://img.shields.io/badge/Powered%20by-Gemma%203-green)](https://ai.google.dev/gemma)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## 🌍 Problem Statement

**263 million children worldwide lack access to quality education.** Many live in low-connectivity regions where traditional online educational tools fail. Teachers often lack resources and training, while students struggle with content that doesn't reflect their cultural context or practical needs.

OfflineLearn bridges this gap by providing an **AI-powered educational assistant that works entirely offline**, adapts to local cultures, and requires minimal resources.

## ✨ What Makes OfflineLearn Different

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
- **Database**: SQLite for offline data storage

## 📦 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 50GB available storage
- 
### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/offlinelearn.git
cd offlinelearn
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create OfflineLearn model:**
```bash
# Create custom model with our educational prompt
ollama create offlinelearn -f Modelfile
```

5. **Initialize the database:**
```bash
python setup_database.py
```

6. **Run the application:**
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

OfflineLearn: "I can see you're working with an equation! Let me ask you this: 
What do you think our goal is when we have an equation like this? What are we 
trying to find?"

[Guides student through Socratic questioning to discover the solution]
```

**Practice Problems:**
```
Request: "Generate 3 fraction problems for grade 6 students in rural Kenya"

OfflineLearn: Creates problems about:
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

## 📁 Project Structure

```
offlinelearn/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 Modelfile                 # Ollama model configuration
├── 🐍 app.py                    # Main Streamlit application
├── 🐍 setup_database.py         # Database initialization
├── 📁 src/
│   ├── 🐍 core/
│   │   ├── 🐍 adaptive_engine.py    # Adaptive learning algorithms
│   │   ├── 🐍 socratic_tutor.py     # Socratic questioning logic
│   │   └── 🐍 cultural_adapter.py   # Cultural context adaptation
│   ├── 🐍 generators/
│   │   ├── 🐍 problem_generator.py  # Dynamic problem creation
│   │   ├── 🐍 lesson_planner.py     # Lesson plan generation
│   │   └── 🐍 assessment_tracker.py # Progress tracking
│   └── 🐍 utils/
│       ├── 🐍 database.py           # Database operations
│       └── 🐍 ollama_client.py      # Ollama API wrapper
├── 📁 data/
│   ├── 📁 training/                 # Fine-tuning datasets
│   ├── 📁 cultural_contexts/        # Regional adaptation data
│   └── 📁 curriculum_standards/     # Educational standards by country
├── 📁 tests/
│   ├── 🐍 test_adaptive_engine.py
│   ├── 🐍 test_problem_generator.py
│   └── 🐍 test_cultural_adapter.py
└── 📁 docs/
    ├── 📄 deployment_guide.md
    ├── 📄 cultural_adaptation.md
    └── 📄 teacher_training.md
```

## 🌟 Core Components



## 🎨 User Interface

### Student Dashboard
- **📊 Progress Overview**: Visual learning progress and achievements
- **📚 Subject Selection**: Easy navigation between subjects
- **🎯 Practice Mode**: Unlimited problems with instant feedback
- **🤝 Tutor Chat**: Conversational learning with AI guide
- **📈 Performance Analytics**: Detailed insights into learning patterns

### Teacher Dashboard
- **👥 Class Management**: Track multiple students' progress
- **📋 Lesson Planning**: Generate and customize lesson plans
- **📊 Analytics**: Class performance insights and recommendations
- **📝 Content Creation**: Create custom exercises and assessments
- **👨‍👩‍👧‍👦 Parent Communication**: Generate progress reports

### Admin Panel
- **⚙️ System Configuration**: Customize for local needs
- **🌏 Cultural Settings**: Adapt interface and content
- **📦 Model Management**: Handle AI model updates
- **📊 Usage Analytics**: System performance monitoring

## 🌐 Cultural Adaptation

OfflineLearn automatically adapts content for different regions:

### Supported Contexts
- **🌾 Rural farming communities** (Kenya, Tanzania, India, etc.)
- **🏪 Trading/market towns** (West Afric
