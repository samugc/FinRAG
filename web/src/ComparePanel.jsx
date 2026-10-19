import { useEffect, useRef, useState } from "react";
import { ArrowRight, CircleAlert, Columns2, FileText, LoaderCircle, X } from "lucide-react";
import "./compare.css";

const RETRIEVERS = [
  { id: "tfidf", label: "TF-IDF", note: "Lexical matching" },
  { id: "minilm", label: "MiniLM", note: "Semantic embeddings" },
];

function keyOf(hit) {
  return JSON.stringify([hit.source, hit.chunk_id]);
}

function SourceCard({ hit, rank, shared }) {
  return (
    <div className="compare-source">
      <div className="compare-source-head">
        <span className="compare-rank">#{rank}</span>
        <span className="compare-source-name"><FileText size={14} /> {hit.source} · chunk {hit.chunk_id}</span>
        {shared && <span className="compare-overlap">Both</span>}
      </div>
      <p>{hit.text}</p>
      <div className="compare-source-score">Similarity score within this method: <b>{Number(hit.score).toFixed(3)}</b></div>
    </div>
  );
}

export default function ComparePanel({ initialQuestion = "", source = null, topK = 3, onClose }) {
  const [question, setQuestion] = useState(initialQuestion);
  const [status, setStatus] = useState("idle");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const controllerRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    function onEscape(event) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, [onClose]);

  async function compare(event) {
    event?.preventDefault();
    const text = question.trim();
    if (!text || status === "loading") return;
    setStatus("loading");
    setError("");
    setData(null);
    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const result = await Promise.all(RETRIEVERS.map(async (method) => {
        const start = performance.now();
        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: text,
            source,
            retriever: method.id,
            generator: "extractive",
            k: topK,
          }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const extra = res.status === 503 ? "The model may still be loading or may be unavailable." : "";
          throw new Error(method.label + " failed (HTTP " + res.status + "). " + extra);
        }
        const payload = await res.json();
        return { ...method, evidence: payload.evidence || [], elapsed: (performance.now() - start) / 1000 };
      }));
      if (mountedRef.current) {
        setData(result);
        setStatus("done");
      }
    } catch (err) {
      controller.abort();
      if (mountedRef.current && err.name !== "AbortError") {
        setError(err.message || "Comparison failed. Check the local API.");
        setStatus("error");
      }
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }

  const keysA = new Set(data?.[0]?.evidence.map(keyOf) || []);
  const keysB = new Set(data?.[1]?.evidence.map(keyOf) || []);
  const overlap = [...keysA].filter((key) => keysB.has(key)).length;

  return (
    <div className="compare-backdrop" role="presentation" onMouseDown={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <section className="compare-modal" role="dialog" aria-modal="true" aria-label="Compare retrieval methods">
        <header className="compare-header">
          <div className="compare-icon"><Columns2 size={21} /></div>
          <div className="compare-heading">
            <span>RETRIEVAL LAB</span>
            <h2>Compare retrievers</h2>
            <p>See what lexical and semantic search retrieve for the same question.</p>
          </div>
          <button className="compare-close" aria-label="Close comparison" onClick={onClose}><X size={20} /></button>
        </header>
        <form className="compare-form" onSubmit={compare}>
          <label htmlFor="compare-question">Question</label>
          <div className="compare-input-row">
            <input id="compare-question" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What was Northstar's operating profit?" maxLength={2000} required />
            <button type="submit" disabled={!question.trim() || status === "loading"}>
              {status === "loading" ? <LoaderCircle className="spin" size={17}/> : <ArrowRight size={17}/>}
              {status === "loading" ? "Comparing…" : "Compare"}
            </button>
          </div>
          <div className="compare-submeta">
            <span>Scope: <b>{source || "All documents"}</b></span>
            <span>Top {topK} per retriever</span>
            <span>Generation: off</span>
          </div>
        </form>

        <div className="compare-body">
          {status === "idle" && <div className="compare-empty">Enter a question to inspect the ranked passages from each retriever. This is a qualitative comparison, not an accuracy benchmark.</div>}
          {status === "loading" && <div className="compare-empty"><LoaderCircle className="spin" size={23}/> Running TF-IDF and MiniLM. The first MiniLM run may download the model.</div>}
          {status === "error" && <div role="alert" className="compare-error"><CircleAlert size={18}/>{error}</div>}
          {data && (
            <>
              <div className="compare-summary">
                <b>{overlap} shared chunk{overlap === 1 ? "" : "s"}</b>
                <span>across the two top-{topK} results · Overlap is not evidence of correctness.</span>
              </div>
              <div className="compare-columns">
                {data.map((method, index) => {
                  const otherKeys = index === 0 ? keysB : keysA;
                  return (
                    <div className="compare-column" key={method.id}>
                      <div className="compare-method-head">
                        <div><h3>{method.label}</h3><span>{method.note}</span></div>
                        <div className="compare-method-meta">{method.evidence.length} chunks</div>
                      </div>
                      {method.evidence.map((hit, rank) =>
                        <SourceCard key={keyOf(hit) + rank} hit={hit} rank={rank + 1} shared={otherKeys.has(keyOf(hit))} />
                      )}
                      {!method.evidence.length && <p className="compare-no-results">No passages returned.</p>}
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
        <footer className="compare-footer">
          <CircleAlert size={16}/>
          <span>Similarity scores from TF-IDF and MiniLM use different representations and must not be compared as if they were calibrated. This view does not calculate benchmark accuracy or test factual grounding.</span>
        </footer>
      </section>
    </div>
  );
}
