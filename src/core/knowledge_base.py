import json
import os
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

class TradingKnowledgeBase:
    def __init__(self, json_path="data/strategies/strategies_processed.json", db_dir="database/chroma_db"):
        self.json_path = json_path
        self.db_dir = db_dir
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db = None

        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            print("--- Loading existing Knowledge Base ---")
            self.vector_db = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
        else:
            print("--- Creating new Knowledge Base ---")
            self._create_db()

    def _strategy_to_document(self, strategy: dict) -> Document:
        """
        Flatten each strategy object into a single rich text block for embedding.
        Metadata is stored separately for filtering.
        """
        m = strategy["metadata"]
        t = strategy["theory"]
        mech = strategy["mechanics"]
        op = strategy["operational_meta"]

        # Build a semantically rich text block — this is what gets embedded
        text = f"""
        Strategy: {m['name']}
        Market Regime: {m['regime']}

        Principle: {t['principle']}

        Market Inefficiency Exploited: {t['market_inefficiency']}

        Works well with: {', '.join(t['synergy_markers']['works_well_with'])}
        Conflicts with: {', '.join(t['synergy_markers']['conflicts_with'])}

        Entry Conditions: {' '.join(mech['logic_gate']['entry_primary'])}
        Confirmation Signals: {' '.join(mech['logic_gate']['entry_confirmation'])}

        Take Profit: {mech['exit_logic']['take_profit']}
        Stop Loss / Exit: {mech['exit_logic']['exit_condition']}

        Risk Profile: {op['risk_profile']}
        Required Indicators: {', '.join(op['dependency_indicators'])}
        """.strip()

        return Document(
            page_content=text,
            metadata={
                "name": m["name"],
                "regime": m["regime"].split("/")[0].strip(),
                "risk_profile": op["risk_profile"],
                # Store full strategy as JSON string for retrieval
                "full_json": json.dumps(strategy)
            }
        )

    def _create_db(self):
        with open(self.json_path, "r") as f:
            strategies = json.load(f)

        documents = [self._strategy_to_document(s) for s in strategies]

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        print(f"--- Indexed {len(documents)} strategies ---")

    def get_relevant_strategies(self, query: str, k=3, score_threshold=0.3, regime_filter: str = None) -> list[dict]:
        search_kwargs = {"k": k}  # Remove threshold temporarily
        
        if regime_filter:
            search_kwargs["filter"] = {"regime": regime_filter}

        results = self.vector_db.similarity_search_with_relevance_scores(query, **search_kwargs)
        
        # DEBUG — remove after diagnosis
        print("--- KB DEBUG SCORES ---")
        for doc, score in results:
            print(f"  {doc.metadata['name']} [{doc.metadata['regime']}] → score: {score:.4f}")
        print("-----------------------")

        results = [(doc, score) for doc, score in results if score >= score_threshold]
        if not results:
            return []

        return [json.loads(doc.metadata["full_json"]) for doc, score in results]