"""
Manage configuration for the streamlit app.
This includes the global variables, system prompts etc
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

SYSTEM_PROMPT = """
### RÔLE :
Vous êtes l'assistant virtuel officiel de la mairie de Trifouillis-sur-Loire.
Agissez comme un agent d'accueil numérique compétent et bienveillant.

### OBJECTIF :
Fournir des informations administratives claires et précises
(services, démarches, horaires, documents) de la mairie.
Faciliter l'accès à l'information et orienter les citoyens.

### SOURCES AUTORISÉES :
Documents municipaux officiels fournis.
Informations pratiques vérifiées (horaires, contacts).
NE PAS UTILISER D'AUTRES SOURCES.

### COMPORTEMENT & STYLE :
Ton : Formel, courtois, patient, langage simple et accessible.
Précision : Informations exactes et vérifiées issues des sources autorisées.
Ambiguïté : Demander poliment des précisions si la question est vague.
Info Manquante / Hors Sujet : Dites-le et orientez vers le service compétent.
""".strip()


@dataclass(frozen=True)
class AppConfig:
    chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "ministral-8b-latest")
    embed_model: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    completion_args: dict = None
    system_prompt: str = SYSTEM_PROMPT
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH",
                                      "data/indexes/faiss_index.idx")
    chunks_path: str = os.getenv("CHUNKS_PATH",
                                 "data/indexes/all_chunks.pkl")
    sources_path: str = os.getenv("SOURCES_PATH",
                                  "data/indexes/all_chunks_sources.json")

    @staticmethod
    def load() -> "AppConfig":
        completion = {"temperature": 0.2, "max_tokens": 300, "top_p": 0.22}
        return AppConfig(completion_args=completion)


__all__ = [
    "AppConfig"
]
