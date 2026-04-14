import json
import requests


class ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments

    @classmethod
    def from_openai(cls, tc):
        return cls(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=json.loads(tc["function"]["arguments"]),
        )


class LLMResponse:
    def __init__(self, content, tool_calls=None, finish_reason="stop"):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason


class LlamaCppBackend:
    def __init__(self, model_path, context_size=4096, n_threads=4, n_gpu_layers=0):
        from llama_cpp import Llama
        print(f"Carregando GGUF: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.context_size = context_size
        print(f"Modelo carregado ({context_size} ctx, {n_threads} threads)")

    def chat(self, messages, system_prompt, temperature=0.7, max_tokens=4096, tools=None):
        full_msgs = [{"role": "system", "content": system_prompt}, *messages]
        kwargs = {
            "messages": full_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        result = self.llm.create_chat_completion(**kwargs)
        choice = result["choices"][0]["message"]
        content = choice.get("content", "")
        tool_calls = []
        if "tool_calls" in choice:
            for tc in choice["tool_calls"]:
                tool_calls.append(ToolCall.from_openai(tc))
        return LLMResponse(content=content, tool_calls=tool_calls)


class OllamaBackend:
    def __init__(self, base_url="http://localhost:11434/v1",
                 model="qwen2.5-coder:3b", api_key="ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def chat(self, messages, system_prompt, temperature=0.7, max_tokens=4096, tools=None):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        content = choice.get("content", "")
        tool_calls = []
        if "tool_calls" in choice:
            for tc in choice["tool_calls"]:
                tool_calls.append(ToolCall.from_openai(tc))
        return LLMResponse(content=content, tool_calls=tool_calls)


def create_backend(model_config):
    if model_config.get("backend") == "llama_cpp":
        return LlamaCppBackend(
            model_path=model_config["model_path"],
            context_size=model_config.get("context_size", 4096),
            n_threads=model_config.get("n_threads", 4),
            n_gpu_layers=model_config.get("n_gpu_layers", 0),
        )
    else:
        return OllamaBackend(
            base_url=model_config.get("base_url", "http://localhost:11434/v1"),
            model=model_config.get("model", "qwen2.5-coder:3b"),
            api_key=model_config.get("api_key", "ollama"),
        )
