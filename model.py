from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,AutoModelForCausalLM

model_path = "google/gemma-3n-E4B-it"
tokenzier = AutoTokenizer.from_pretrained(model_path)

prompt = "It was a dark and stormy night."

tokenized_prompt = tokenzier(prompt, return_tensor="pt",padding=True, truncation=True)["input_ids"]

print("Tokenized prompt is: ",tokenized_prompt)


# NEED TO USE LLM WAY RATHER THAN AGENT WAY


# problem_gen_agent = LlmAgent(
#     model=LiteLlm(model="gemma3n:e4b"),
#     name="dice_agent",
#     description=(
# """
# You are great at generating contextually relevant practice problems.
# Like math, science, language arts, and vocational training all in one system.
# """

#     ),
#     instruction=    """Generate contextually relevant practice problems:
#     prompt =Create 5 {subject} problems at difficulty level {difficulty_level}/10.
#     Use local context: {local_context} (e.g., farming, local economy, cultural references)
    
#     For each problem provide:
#     1. Problem statement with local examples
#     2. Step-by-step solution
#     3. Common mistakes to avoid
#     4. Real-world application
#     """,
#     tools=[],
# )

# problem_guide_agent = LlmAgent(
#     model=LiteLlm(model="gemma3n:e4b"),
#     name="dice_agent",
#     description=(
# """You are great at guiding students through problems with questions.
# Like math, science, language arts, and vocational training all in one system.
# """
#     ),
#     instruction=     """Guide students through problems with questions
#     prompt = Student asked: "{student_question}" in {subject_area}
    
#     Don't give the answer directly. Instead:
#     1. Ask a guiding question that helps them think
#     2. Provide a small hint about the approach
#     3. Encourage them to try the next step
#     4. If they're stuck, ask what they understand so far
#     """,
#     tools=[],
# )
# lesson_plan_agent = LlmAgent(
#     model=LiteLlm(model="gemma3n:e4b"),
#     name="lesson planner",
#     description=(
# """ 
# You are great at generation offline lesson plans for students.
# Like math, science, language arts, and vocational training all in one system.

# """
#     ),
#     instruction=       """Generate complete offline lesson plans
#     prompt = Create a {available_time}-minute lesson plan for {topic} at grade {grade_level}.
    
#     Include:
#     1. Learning objectives
#     2. Materials needed (assuming limited resources)
#     3. Step-by-step activities
#     4. Assessment questions
#     5. Extension activities for advanced students
#     6. Remediation for struggling students
#     """,
#     tools=[],
# )
# extra_less_agent = LlmAgent(
#     model=LiteLlm(model="gemma3n:e4b"),
#     name="dice_agent",
#     description=(
# """
# You are great at generation offline extra curricular plans for students.Like imporoving their communication skills,
# by adding some basic to intermediate to advance level communicaution activities and lesson that student will feel free and willing to practice with you.
# """
#     ),
#     instruction=       """Generate complete offline lesson plans
#     prompt = Create a {available_time}-minute lesson plan for extra curricular activities like communication skill at basic to intermediate to advance levle.
    
#     Include:
#     1. Learning objectives
#     2. Materials needed (assuming limited resources)
#     3. Step-by-step activities
#     4. Assessment questions
#     5. Extension activities for advanced students
#     6. Remediation for struggling students
#     """,
#     tools=[],
# )

# main_agent = SequentialAgent(
#     name="CodePipelineAgent",
#     sub_agents=[problem_gen_agent,problem_guide_agent,lesson_plan_agent,extra_less_agent],
#     description="Executes a sequence of code writing, reviewing, and refactoring.",
#     # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
# )

# # For ADK tools compatibility, the root agent must be named `root_agent`
# root_agent = main_agent