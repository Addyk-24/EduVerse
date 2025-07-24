from google.adk.agents import LlmAgent
from google.genai import types # For config objects
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent, SequentialAgent

from run_ollama import chat
from run_ollama import ChatResponse

# can use
from unsloth import FastModel
import torch
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # Or "unsloth/gemma-3n-E2B-it"
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
from transformers import TextStreamer
import gc
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()

# ---- x -----
model = "gemma3n:e4b"




def Document_parser():
    pass

class NeuroMirror:
    def __init__(self,model):
        self.model = model
    
    def chat():
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},

        ]
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("Bye I hope you learned something new today!")
                break
            else:
                response : ChatResponse = chat(
                    model

                )

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