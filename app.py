from google.adk.agents import LlmAgent
from google.genai import types # For config objects
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent, SequentialAgent

from run_ollama import chat
from run_ollama import ChatResponse

# can use
import torch

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
    
    def chat(self):
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
                    model = model,


                )

    device_count = torch.cuda.device_count()
    if device_count > 0:
        logger.debug("Select GPU device")
        device = torch.device("cuda")
    else:
        logger.debug("Select CPU device")
        device = torch.device("cpu")


    def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
        #Tokenize
        input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
        )
        # Generate
        device = model.device
        generated_tokens_with_prompt = model.generate(
            input_ids=input_ids.to(device),
            max_length=max_output_tokens
        )
        # Decode
        generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

        # Strip the prompt
        generated_text_answer = generated_text_with_prompt[0][len(text):]

        return generated_text_answer

