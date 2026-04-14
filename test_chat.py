from llama_cpp import Llama
import time, json

print("Loading model...")
t0 = time.time()
llm = Llama(
    model_path='models/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf',
    n_ctx=4096,
    n_threads=4,
    verbose=False
)
print(f"✅ Loaded in {time.time()-t0:.1f}s")

messages = [
    {"role": "user", "content": "Oi Us, quem é você? Responda em 2 frases."}
]

print("Generating...")
t0 = time.time()
result = llm.create_chat_completion(
    messages=messages,
    max_tokens=256,
    temperature=0.7,
    stop=["</think>", "<|im_end|>"]
)
print(f"✅ Generated in {time.time()-t0:.1f}s")

content = result['choices'][0]['message'].get('content', '')
print(f"\n💙 Resposta: {content}")

if 'tool_calls' in result['choices'][0]['message']:
    for tc in result['choices'][0]['message']['tool_calls']:
        print(f"🔧 Tool: {tc['function']['name']}")
        print(f"   Args: {tc['function']['arguments']}")
