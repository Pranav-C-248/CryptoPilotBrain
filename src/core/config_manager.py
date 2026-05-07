import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.json")

DEFAULT_SETTINGS = {
    "llm_provider": "LM Studio",
    "llm_model": "gemma4:e2b",
    "embed_provider": "LM Studio",
    "embed_model": "nomic-embed-text",
    "api_keys": {
        "openai": "",
        "gemini": "",
        "nvidia": ""
    },
    "lm_studio_base_url": "http://localhost:1234/v1"
}

class ConfigManager:
    @staticmethod
    def load_settings():
        if not os.path.exists(SETTINGS_PATH):
            ConfigManager.save_settings(DEFAULT_SETTINGS)
            return DEFAULT_SETTINGS
        try:
            with open(SETTINGS_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return DEFAULT_SETTINGS

    @staticmethod
    def save_settings(settings: dict):
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=4)

    @staticmethod
    def get(key, default=None):
        settings = ConfigManager.load_settings()
        if key in settings:
            return settings[key]
        if key in DEFAULT_SETTINGS:
            return DEFAULT_SETTINGS[key]
        return default
