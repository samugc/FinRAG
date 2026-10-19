import { useEffect, useRef, useState } from "react";
import ComparePanel from "./ComparePanel.jsx";
import {
  ArrowDownRight, ArrowRight, BookOpen, Check, ChevronDown, ChevronRight,
  CircleAlert, CircleCheck, Clock3, Command, Database, FileText, Filter,
  Layers3, LoaderCircle, Menu, MessageSquareText, PanelRightClose,
  RotateCcw, Search, Send, Settings2, ShieldCheck, Sparkles, X, Columns2, Upload,
} from "lucide-react";

const SAMPLES = [
  "What was Northstar's operating profit?",
  "How much cash remained after investment spending?",
  "Which currencies create exchange-rate exposure?",
];
const niceName = (name) => name.replace(/\.(txt|md|pdf)$/i, "").replace(/[_-]/g, " ");
function labelName(name) {
  return niceName(name).replace(/\b\w/g, (c) => c.toUpperCase());
}
function shortText(value = "", max = 165) {
  return value.length > max ? value.slice(0, max).trimEnd() + "…" : value;
}
function localMessage(error) {
  if (error?.name === "AbortError") return "Request timed out. The model may need more time to initialize.";
  return error?.message || "The request could not be completed.";
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [health, setHealth] = useState(null);
  const [online, setOnline] = useState(false);
  const [loadingSetup, setLoadingSetup] = useState(true);
  const [selectedDoc, setSelectedDoc] = useState("all");
  const [retriever, setRetriever] = useState("tfidf");
  const [generator, setGenerator] = useState("extractive");
  const [k, setK] = useState(3);
  const [query, setQuery] = useState("");
  const [entries, setEntries] = useState([]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [activeEvidence, setActiveEvidence] = useState(0);
  const [mobilePanel, setMobilePanel] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const fileInputRef = useRef(null);
  const abortRef = useRef(null);
  const bottomRef = useRef(null);

  async function refresh() {
    setLoadingSetup(true);
    try {
      const [healthRes, docsRes] = await Promise.all([
        fetch("/api/health"), fetch("/api/documents"),
      ]);
      if (!healthRes.ok || !docsRes.ok) throw new Error("API unavailable");
      const [h, d] = await Promise.all([healthRes.json(), docsRes.json()]);
      if (h.status !== "ok") throw new Error("Corpus unavailable");
      setHealth(h);
      setDocuments(d.documents || []);
      setOnline(true);
      setError("");
    } catch {
      setOnline(false);
      setHealth(null);
      setDocuments([]);
    } finally {
      setLoadingSetup(false);
    }
  }

  useEffect(() => { refresh(); return () => abortRef.current?.abort(); }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [entries, working]);

  const last = [...entries].reverse().find((entry) => entry.type === "answer");
  const evidence = last?.data?.evidence || [];
  const visibleEvidence = evidence;

  async function ask(text = query) {
    const question = text.trim();
    if (!question || working) return;
    if (!online) { setError("Connect to the local API before asking a question."); return; }
    setWorking(true);
    setError("");
    setEntries((prev) => [...prev, { type: "question", text: question }]);
    setQuery("");
    setActiveEvidence(0);
    const controller = new AbortController();
    abortRef.current = controller;
    const start = performance.now();
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, retriever, generator, k, source: selectedDoc === "all" ? null : selectedDoc }),
        signal: controller.signal,
      });
      if (!res.ok) {
        let data = {};
        try { data = await res.json(); } catch { /* non-JSON response */ }
        throw new Error(res.status === 503 ? "The model or document corpus is unavailable." : "Query failed (HTTP " + res.status + "). " + (typeof data.detail === "string" ? data.detail : ""));
      }
      const data = await res.json();
      setEntries((prev) => [...prev, { type: "answer", data, elapsed: (performance.now() - start) / 1000 }]);
    } catch (err) {
      if (err.name !== "AbortError") {
        setError(localMessage(err));
        setEntries((prev) => [...prev, { type: "error", text: localMessage(err) }]);
      }
    } finally {
      setWorking(false);
      abortRef.current = null;
    }
  }

  async function uploadPdf(file) {
    if (!file || uploading) return;
    if (!/\.pdf$/i.test(file.name) || file.size > 25 * 1024 * 1024) {
      setError("Choose a PDF smaller than 25 MB.");
      return;
    }
    setUploading(true);
    setUploadMessage("");
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/api/documents", {
        method: "POST",
        headers: { "X-FinRAG-Upload": "1" },
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed");
      setSelectedDoc(data.source);
      setUploadMessage("Added " + data.source + " · " + data.chunks + " chunks");
      await refresh();
    } catch (err) {
      setError(err.message || "Unable to upload PDF.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  const hasStarted = entries.length > 0;
  const body = (
    <>
      {!hasStarted && (
        <div className="intro">
          <div className="intro-icon"><Sparkles size={21} strokeWidth={1.8} /></div>
          <div className="eyebrow">YOUR RESEARCH WORKSPACE</div>
          <h1>Ask better questions.<br /><span>Find the evidence.</span></h1>
          <p>Explore financial reports through retrieval-grounded answers. Every response comes with the source material so you can inspect it yourself.</p>
          <div className="suggestions">
            <div className="suggest-title">TRY ASKING {selectedDoc !== "all" ? "· SELECTED DOCUMENT" : ""}</div>
            {SAMPLES.map((item) => (
              <button key={item} className="suggestion" onClick={() => ask(item)}>
                <span>{item}</span><ArrowRight size={16} />
              </button>
            ))}
          </div>
        </div>
      )}
      {hasStarted && (
        <div className="conversation">
          {entries.map((entry, index) => entry.type === "question" ? (
            <div className="message-question" key={index}>{entry.text}</div>
          ) : entry.type === "error" ? (
            <div className="message-error" key={index}><CircleAlert size={17} /> {entry.text}</div>
          ) : (
            <div className="message-answer" key={index}>
              <div className="assistant-mark"><Sparkles size={16}/></div>
              <div className="answer-content">
                <div className="answer-head"><b>FinRAG</b><span>{entry.data.retriever.toUpperCase()} · {entry.data.generator === "extractive" ? "Extractive" : "Qwen"}</span></div>
                <div className="answer-text">{entry.data.answer}</div>
                <div className="answer-footer">
                  <span className={entry.data.sufficient_evidence ? "grounding caution" : "grounding insufficient"}>
                    {entry.data.sufficient_evidence ? <CircleCheck size={14}/> : <CircleAlert size={14}/>}
                    {entry.data.sufficient_evidence ? "Evidence retrieved · not fact-verified" : "Insufficient evidence"}
                  </span>
                  <button className="source-link" onClick={() => { setMobilePanel("evidence"); setActiveEvidence(0); }}>
                    <BookOpen size={14}/> {entry.data.evidence.length} sources <ArrowRight size={13}/>
                  </button>
                  <span className="time"><Clock3 size={13}/> {entry.elapsed.toFixed(1)}s</span>
                </div>
              </div>
            </div>
          ))}
          {working && (
            <div className="thinking"><span className="assistant-mark"><LoaderCircle size={16} className="spin"/></span><span>Retrieving documents{generator === "qwen" ? " and generating a response" : ""}…</span></div>
          )}
          <div ref={bottomRef}/>
        </div>
      )}
    </>
  );

  return (
    <div className="app">
      <aside className={"sidebar " + (mobilePanel === "documents" ? "mobile-show" : "")}>
        <div className="brand"><div className="brand-icon"><Layers3 size={21}/></div><span>finrag<span className="brand-dot">.</span></span><button className="mobile-close" onClick={() => setMobilePanel("")}><X size={19}/></button></div>
        <div className="sidebar-inner">
          <div className="section-heading"><span>WORKSPACE</span></div>
          <button className="nav-link nav-active" onClick={() => setMobilePanel("")}><MessageSquareText size={18}/> Research assistant <ChevronRight size={15}/></button>
          <div className="section-heading source-heading"><span>DOCUMENT LIBRARY</span><span className="number">{documents.length}</span></div>
          <button className={"document-item " + (selectedDoc === "all" ? "doc-selected" : "")} onClick={() => { setSelectedDoc("all"); setMobilePanel(""); }}>
            <Database size={17}/><span>All documents</span>
          </button>
          {documents.map((doc) => (
            <button className={"document-item " + (selectedDoc === doc.source ? "doc-selected" : "")} key={doc.source} onClick={() => { setSelectedDoc(doc.source); setMobilePanel(""); }}>
              <FileText size={17}/><span title={doc.source}>{labelName(doc.source)}</span>
            </button>
          ))}
          {!documents.length && !loadingSetup && <div className="doc-empty">No documents available</div>}
          <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" aria-label="Select a PDF document" className="pdf-file-input" onChange={(e) => uploadPdf(e.target.files?.[0])} />
          <button className="upload-document" onClick={() => fileInputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); uploadPdf(e.dataTransfer.files?.[0]); }} disabled={uploading || working || !online} title="Click or drop a PDF file here">
            {uploading ? <LoaderCircle size={16} className="spin"/> : <Upload size={16}/>}
            {uploading ? "Adding PDF…" : "Upload or drop PDF"}
          </button>
          {uploadMessage && <div className="upload-message" role="status">{uploadMessage}</div>}
          <div className="sidebar-note"><ShieldCheck size={17}/><div><b>Local-first research</b><span>Documents stay on your machine. No paid API required.</span></div></div>
        </div>
        <div className="sidebar-bottom"><span className="avatar">FR</span><div><b>FinRAG Workspace</b><span>Research environment</span></div><Settings2 size={17}/></div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="top-left"><button className="icon-button mobile-menu" aria-label="Open documents" onClick={() => setMobilePanel("documents")}><Menu size={20}/></button><span>Workspace</span><ChevronRight size={14}/><b>Research assistant</b></div>
          <div className="top-right">
            <button className="compare-launch" onClick={() => setCompareOpen(true)} disabled={!online} title="Compare TF-IDF and MiniLM on the same question"><Columns2 size={16}/> Compare</button>
            <button className="connection" title="Retry API connection" onClick={refresh}>
              <span className={"dot " + (online ? "online" : "offline")}/>{loadingSetup ? "Checking…" : online ? "API connected" : "API offline"} <RotateCcw size={13}/>
            </button>
            <button className="right-toggle" onClick={() => setMobilePanel("evidence")}><PanelRightClose size={19}/></button>
          </div>
        </header>
        <div className="main-scroll">
          {body}
        </div>
        <div className="composer-area">
          {!!error && <div className="global-error"><CircleAlert size={15}/>{error}<button aria-label="Dismiss error" onClick={() => setError("")}><X size={15}/></button></div>}
          <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(); }}>
            <textarea aria-label="Ask a financial question" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }} placeholder="Ask anything about your financial documents..." rows={2}/>
            <div className="composer-actions">
              <div className="options">
                <label className="model-select"><Search size={15}/><select aria-label="Retriever" value={retriever} onChange={(e) => setRetriever(e.target.value)}><option value="tfidf">TF-IDF</option><option value="minilm">MiniLM</option></select><ChevronDown size={12}/></label>
                <label className="model-select generator"><Sparkles size={15}/><select aria-label="Generator" value={generator} onChange={(e) => setGenerator(e.target.value)}><option value="extractive">Extractive</option><option value="qwen">Qwen local</option></select><ChevronDown size={12}/></label>
                <button type="button" className={"setting-toggle " + (settingsOpen ? "setting-on" : "")} onClick={() => setSettingsOpen(!settingsOpen)}><Filter size={15}/> Settings</button>
              </div>
              <button aria-label="Send question" className="send-btn" type="submit" disabled={!query.trim() || working || !online}>{working ? <LoaderCircle className="spin" size={19}/> : <ArrowRight size={19}/>}</button>
            </div>
            {settingsOpen && <div className="settings-row"><label>Evidence chunks <b>{k}</b></label><input aria-label="Evidence chunks" type="range" min="1" max="10" value={k} onChange={(e)=>setK(Number(e.target.value))}/><span>Top {k}</span></div>}
          </form>
          <div className="composer-footnote">AI-generated answers can be incorrect. Always verify against the cited source text.</div>
        </div>
      </main>

      <aside className={"evidence-panel " + (mobilePanel === "evidence" ? "mobile-show" : "")}>
        <div className="evidence-title"><div><BookOpen size={18}/><b>Evidence explorer</b></div><button className="mobile-close" onClick={() => setMobilePanel("")}><X size={19}/></button></div>
        <div className="evidence-content">
          <div className="evidence-label">RETRIEVED SOURCES <span>{visibleEvidence.length}</span></div>
          {!evidence.length ? (
            <div className="evidence-placeholder"><div className="evidence-placeholder-icon"><FileText size={23}/></div><b>Your sources appear here</b><p>Ask a question to explore retrieved passages, relevance scores, and document origins.</p></div>
          ) : (
            <>
              {selectedDoc !== "all" && <div className="filter-note"><Filter size={13}/> Scoped to {labelName(selectedDoc)} <button onClick={() => setSelectedDoc("all")}>Clear</button></div>}
              {visibleEvidence.map((hit, idx) => (
                <button key={hit.source + hit.chunk_id + idx} className={"evidence-card " + (activeEvidence === idx ? "active-evidence" : "")} onClick={() => setActiveEvidence(idx)}>
                  <div className="evidence-card-top"><div className="file-badge"><FileText size={16}/></div><div className="file-head"><b>{labelName(hit.source)}</b><span>Chunk {hit.chunk_id} · Rank {evidence.indexOf(hit) + 1}</span></div><ChevronRight size={15}/></div>
                  <p>{activeEvidence === idx ? hit.text : shortText(hit.text)}</p>
                  <div className="evidence-card-bottom"><span><span className="score-dot"/> Relevance score</span><b>{Number(hit.score).toFixed(3)}</b></div>
                  <div className="score-track"><div style={{width: Math.min(100, Math.max(0, Number(hit.score) * 100)) + "%"}}/></div>
                </button>
              ))}
              {!visibleEvidence.length && <div className="doc-empty">No retrieved sources match this document filter.</div>}
            </>
          )}
        </div>
        <div className="evidence-footer"><CircleAlert size={16}/><span>Scores rank passages; they do not prove that an answer is correct.</span></div>
      </aside>
      {compareOpen && <ComparePanel initialQuestion={query || [...entries].reverse().find(entry => entry.type === "question")?.text || ""} source={selectedDoc === "all" ? null : selectedDoc} topK={k} onClose={() => setCompareOpen(false)} />}
      {mobilePanel && <button className="mobile-overlay" aria-label="Close panel" onClick={() => setMobilePanel("")}/>}
    </div>
  );
}
