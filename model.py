from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,AutoModelForCausalLM

model_path = "google/gemma-3n-E4B-it"
tokenzier = AutoTokenizer.from_pretrained(model_path)

prompt = "It was a dark and stormy night."

tokenized_prompt = tokenzier(prompt, return_tensor="pt",padding=True, truncation=True)["input_ids"]

print("Tokenized prompt is: ",tokenized_prompt)


