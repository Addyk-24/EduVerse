from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3n', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
print(response.message.content)