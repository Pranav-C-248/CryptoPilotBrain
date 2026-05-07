import os
from src.core.config_manager import ConfigManager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class LLMFactory:
    @staticmethod
    def get_llm():
        settings = ConfigManager.load_settings()
        provider = settings.get("llm_provider", "LM Studio")
        model = settings.get("llm_model", "gemma4:e2b")
        keys = settings.get("api_keys", {})
        
        if provider == "LM Studio":
            base_url = settings.get("lm_studio_base_url", "http://localhost:1234/v1")
            return ChatOpenAI(
                base_url=base_url,
                api_key="lm-studio",
                model=model,
                temperature=0
            )
        elif provider == "OpenAI":
            api_key = keys.get("openai", os.getenv("OPENAI_API_KEY", ""))
            return ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0
            )
        elif provider == "Gemini":
            api_key = keys.get("gemini", os.getenv("GEMINI_API_KEY", ""))
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                temperature=0
            )
        elif provider == "NVIDIA":
            api_key = keys.get("nvidia", os.getenv("NVIDIA_API_KEY", ""))
            return ChatOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                model=model,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_embeddings():
        settings = ConfigManager.load_settings()
        provider = settings.get("embed_provider", "LM Studio")
        model = settings.get("embed_model", "nomic-embed-text")
        keys = settings.get("api_keys", {})
        
        if provider == "LM Studio":
            base_url = settings.get("lm_studio_base_url", "http://localhost:1234/v1")
            return OpenAIEmbeddings(
                base_url=base_url,
                api_key="lm-studio",
                model=model,
                check_embedding_ctx_length=False
            )
        elif provider == "OpenAI":
            api_key = keys.get("openai", os.getenv("OPENAI_API_KEY", ""))
            return OpenAIEmbeddings(
                api_key=api_key,
                model=model
            )
        elif provider == "Gemini":
            api_key = keys.get("gemini", os.getenv("GEMINI_API_KEY", ""))
            return GoogleGenerativeAIEmbeddings(
                google_api_key=api_key,
                model=model
            )
        elif provider == "NVIDIA":
            api_key = keys.get("nvidia", os.getenv("NVIDIA_API_KEY", ""))
            return OpenAIEmbeddings(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                model=model
            )
        else:
            raise ValueError(f"Unsupported Embeddings provider: {provider}")
