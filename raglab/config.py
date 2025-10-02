from pydantic import BaseModel
from pathlib import Path
import yaml


class Settings(BaseModel):
    project_name: str
    seed_domain: str
    seeds_file: str
    scrape: dict
    chunk: dict
    embeddings: dict
    index: dict
    retriever: dict
    llm: dict
    ui: dict


    @staticmethod
    def load(path: str) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return Settings(**cfg)