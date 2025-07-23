# Example functions you can build:

def generate_adaptive_problems(subject, difficulty_level, local_context):
    """Generate contextually relevant practice problems"""
    prompt = f"""
    Create 5 {subject} problems at difficulty level {difficulty_level}/10.
    Use local context: {local_context} (e.g., farming, local economy, cultural references)
    
    For each problem provide:
    1. Problem statement with local examples
    2. Step-by-step solution
    3. Common mistakes to avoid
    4. Real-world application
    """

def socratic_tutor(student_question, subject_area):
    """Guide students through problems with questions"""
    prompt = f"""
    Student asked: "{student_question}" in {subject_area}
    
    Don't give the answer directly. Instead:
    1. Ask a guiding question that helps them think
    2. Provide a small hint about the approach
    3. Encourage them to try the next step
    4. If they're stuck, ask what they understand so far
    """

def create_lesson_plan(topic, grade_level, available_time):
    """Generate complete offline lesson plans"""
    prompt = f"""
    Create a {available_time}-minute lesson plan for {topic} at grade {grade_level}.
    
    Include:
    1. Learning objectives
    2. Materials needed (assuming limited resources)
    3. Step-by-step activities
    4. Assessment questions
    5. Extension activities for advanced students
    6. Remediation for struggling students
    """

# Core Innovation:

# Adaptive Learning Engine: Uses Gemma 3n to analyze student responses and dynamically adjust difficulty/teaching style
# Contextual Content Generation: Creates unlimited practice problems, explanations, and examples based on local context
# Socratic Teaching Method: Guides students through problems with questions rather than giving direct answers
# Multi-Subject Support: Math, science, language arts, and vocational training all in one system
# Progress Tracking: Identifies knowledge gaps and creates personalized study plans