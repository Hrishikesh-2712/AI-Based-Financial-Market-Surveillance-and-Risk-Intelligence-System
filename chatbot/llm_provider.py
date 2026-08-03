"""
llm_provider.py
================
Factory that returns a chat LLM based on config.LLM_PROVIDER, so the
rest of the app never needs to know whether it's talking to Gemini
(cloud, needs an API key) or Ollama (fully local, no key needed).
"""

from . import config


def get_llm():
    if config.LLM_PROVIDER == "gemini":
        if not config.GOOGLE_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is not set. "
                "Add it to your .env file, or set LLM_PROVIDER=ollama to "
                "run fully locally instead."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.2,
        )

    elif config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'gemini' or 'ollama'."
        )
