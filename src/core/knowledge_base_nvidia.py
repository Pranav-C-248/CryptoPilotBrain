import json
import os
import time
import shutil
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma


# ── Prompt-payload builder ────────────────────────────────────────────────────
# Separate function so it can be unit-tested independently of ChromaDB.
# This is what the LLM sees — concise, structured, token-efficient.
#
# Excluded from prompt payload (lives in embedding text only):
#   theory.principle / market_inefficiency  → prose, aids retrieval not evaluation
#   theory.synergy_markers                  → human labels, LLM uses structured fields
#   exit_logic.take_profit                  → informational, levels set by RiskManager
#   operational_meta.dependency_indicators  → redundant with structured conditions

def _build_prompt_payload(strategy: dict) -> dict:
    m    = strategy["metadata"]
    mech = strategy["mechanics"]
    gate = mech["logic_gate"]
    op   = strategy["operational_meta"]

    return {
        "name":                          m["name"],
        "regime":                        m["regime"],      # full {primary, also_valid, forbidden} object
        "risk_profile":                  op["risk_profile"],
        "entry_primary":                 gate["entry_primary"],
        "entry_primary_structured":      gate["entry_primary_structured"],
        "entry_confirmation":            gate["entry_confirmation"],
        "entry_confirmation_structured": gate["entry_confirmation_structured"],
        "conflicts_structured":          gate["conflicts_structured"],
        "exit_condition":                mech["exit_logic"]["exit_condition"],
        "exit_condition_structured":     mech["exit_logic"]["exit_condition_structured"],
    }


class TradingKnowledgeBase:
    """
    Vector-backed strategy library using ChromaDB + nomic-embed-text (via LM Studio).

    Design decisions
    ────────────────
    Embedding text    — rich prose from theory + human-readable entry/exit descriptions.
                        Semantic richness here drives retrieval quality.
    ChromaDB metadata — stores flat regime strings for $or/$contains filtering,
                        plus a lean prompt_payload JSON for injection. The full
                        strategy JSON is NOT stored in metadata to avoid token bloat.
    Regime filter     — multi-value: matches primary AND also_valid regimes. Strategies
                        in the forbidden list are removed via post-filter (ChromaDB
                        does not support "not contains" on string metadata).
    Score threshold   — applied as both a fetch-time inflation (fetch_k = k*3) and a
                        post-filter, so callers reliably get up to k results back.
    Prompt payload    — stripped-down dict; excludes all prose the LLM does not evaluate.
    """

    def __init__(
        self,
        json_path: str = "data/strategies/strategies_processed.json",
        db_dir: str    = "database/chroma_db_nvidia",
    ):
        self.json_path = json_path
        self.db_dir    = db_dir

        self.embeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key="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",
            truncate="END",
        )
        self.vector_db = None

        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            print("--- Loading existing Knowledge Base ---")
            self.vector_db = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
            )
        else:
            print("--- Creating new Knowledge Base ---")
            self._create_db()

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _strategy_to_document(self, strategy: dict) -> Document:
        """
        Build the ChromaDB Document for one strategy.

        page_content  → rich prose text, optimised for semantic embedding.
        metadata      → flat scalars used for filtering + lean prompt_payload JSON.

        ChromaDB metadata only supports scalar types (str, int, float, bool).
        The regime object's lists are serialized as comma-separated strings so
        ChromaDB's $contains operator can match substrings for also_valid lookups.
        """
        m    = strategy["metadata"]
        t    = strategy["theory"]
        mech = strategy["mechanics"]
        op   = strategy["operational_meta"]
        gate = mech["logic_gate"]

        regime: dict = m["regime"]   # {primary, also_valid, forbidden}

        # ── Regime description for embedding text ─────────────────────────────
        also_valid_str = (
            f" (also valid in: {', '.join(regime['also_valid'])})"
            if regime.get("also_valid")
            else ""
        )
        forbidden_str = (
            f" (forbidden in: {', '.join(regime['forbidden'])})"
            if regime.get("forbidden")
            else ""
        )

        # Filter nulls before joining — some structured conditions are null
        # for conditions that cannot be mechanically expressed
        entry_primary_text = " ".join(
            x for x in gate["entry_primary"] if x is not None
        )
        entry_confirm_text = " ".join(
            x for x in gate["entry_confirmation"] if x is not None
        )

        # ── Embedding text ────────────────────────────────────────────────────
        text = f"""
Strategy: {m['name']}
Market Regime: {regime['primary']}{also_valid_str}{forbidden_str}

Principle: {t['principle']}

Market Inefficiency Exploited: {t['market_inefficiency']}

Works well with: {', '.join(t['synergy_markers']['works_well_with'])}
Conflicts with: {', '.join(t['synergy_markers']['conflicts_with'])}

Entry Conditions: {entry_primary_text}
Confirmation Signals: {entry_confirm_text}

Take Profit: {mech['exit_logic']['take_profit']}
Stop Loss / Exit: {mech['exit_logic']['exit_condition']}

Risk Profile: {op['risk_profile']}
Required Indicators: {', '.join(op['dependency_indicators'])}
        """.strip()

        # ── ChromaDB metadata ─────────────────────────────────────────────────
        also_valid_csv = ",".join(regime.get("also_valid", []))
        forbidden_csv  = ",".join(regime.get("forbidden", []))

        return Document(
            page_content=text,
            metadata={
                "name":              m["name"],
                "regime_primary":    regime["primary"],
                "regime_also_valid": also_valid_csv,   # e.g. "Trending,Volatile" or ""
                "regime_forbidden":  forbidden_csv,    # e.g. "Sideways" or ""
                "risk_profile":      op["risk_profile"],
                "prompt_payload":    json.dumps(_build_prompt_payload(strategy)),
            },
        )

    def _create_db(self):
        with open(self.json_path, "r") as f:
            strategies = json.load(f)

        documents = [self._strategy_to_document(s) for s in strategies]

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_dir,
        )
        print(f"--- Indexed {len(documents)} strategies into Knowledge Base ---")

    def rebuild(self):
        """
        Force a full rebuild of the ChromaDB store from the source JSON.
        Call this after updating Strategies_v2.json.
        """
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
            print(f"--- Cleared existing KB at {self.db_dir} ---")
        self._create_db()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _build_regime_filter(self, regime: str | None) -> dict | None:
        """
        Build a ChromaDB $or filter that matches strategies where:
          - regime_primary == regime, OR
          - regime_also_valid contains regime (substring match on CSV string)

        Returns None when regime is None (undetected market structure).
        The caller will then run an unfiltered search and rely solely on
        the score threshold and forbidden post-filter.
        """
        if regime is None:
            return None

        return {
            "$or": [
                {"regime_primary":    {"$eq": regime}},
                {"regime_also_valid": {"$contains": regime}},
            ]
        }

    def _is_forbidden(self, doc_metadata: dict, regime: str | None) -> bool:
        """
        Return True if the current regime appears in this strategy's forbidden list.
        Post-filter only — ChromaDB cannot do "not contains" on string metadata.
        """
        if not regime:
            return False
        forbidden_csv = doc_metadata.get("regime_forbidden", "")
        if not forbidden_csv:
            return False
        forbidden = [r.strip() for r in forbidden_csv.split(",")]
        return regime in forbidden

    def get_relevant_strategies(
        self,
        query: str,
        k: int = 2,
        score_threshold: float = 0.40,
        regime_filter: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the top-k strategies most semantically similar to query,
        filtered by regime compatibility and score threshold.

        Parameters
        ----------
        query           : Market state description built by _build_query()
        k               : Maximum strategies to return after all filtering
        score_threshold : Minimum relevance score (0.0–1.0)
        regime_filter   : Current regime string or None

        Returns
        -------
        List of lean prompt payload dicts — not full strategy JSON.
        Empty list if no results pass the threshold/regime filters.
        """
        # Inflate fetch count to absorb expected post-filter losses
        # (forbidden strategies + below-threshold results).
        fetch_k = k * 3

        chroma_filter = self._build_regime_filter(regime_filter)

        attempt = 0
        base_delay = 2
        max_delay = 60
        results = []

        while True:
            try:
                if chroma_filter:
                    results = self.vector_db.similarity_search_with_relevance_scores(
                        query, k=fetch_k, filter=chroma_filter
                    )
                else:
                    # No regime detected — unfiltered search, forbidden post-filter still runs
                    results = self.vector_db.similarity_search_with_relevance_scores(
                        query, k=fetch_k
                    )
                break
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str or "rate limit" in error_str:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    print(f"[KB] NVIDIA API Rate Limit (429). Retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                print(f"[KB] ChromaDB search failed: {e}. Returning empty.")
                return []

        # ── Debug log ─────────────────────────────────────────────────────────
        print("--- KB RETRIEVAL ---")
        for doc, score in results:
            meta           = doc.metadata
            forbidden_flag = " ⛔ FORBIDDEN" if self._is_forbidden(meta, regime_filter) else ""
            pass_flag      = "✓" if score >= score_threshold else "✗"
            print(
                f"  [{pass_flag}] {meta['name']}"
                f" | primary={meta['regime_primary']}"
                f" | also_valid=[{meta['regime_also_valid'] or 'none'}]"
                f" | score={score:.4f}"
                f"{forbidden_flag}"
            )
        print("--------------------")

        # ── Post-filters ──────────────────────────────────────────────────────
        filtered = [
            (doc, score)
            for doc, score in results
            if score >= score_threshold
            and not self._is_forbidden(doc.metadata, regime_filter)
        ]

        if not filtered:
            print(
                f"[KB] No strategies passed filters "
                f"(regime={regime_filter}, threshold={score_threshold})."
            )
            return []

        # Sort descending by score, cap at k
        filtered.sort(key=lambda x: x[1], reverse=True)
        top = filtered[:k]

        return [json.loads(doc.metadata["prompt_payload"]) for doc, _ in top]