import { useState, useCallback } from "react";
import { Search, GitCompare, Globe, ExternalLink, Loader2, AlertTriangle, Zap } from "lucide-react";

const DEFAULT_THRESHOLD = 0.85;

function SimilarityBar({ value, threshold = DEFAULT_THRESHOLD }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const isMatch = value >= threshold;
  return (
    <div className="w-full">
      <div className="relative h-2 rounded-full bg-[#1B2437] overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background: isMatch
              ? "linear-gradient(90deg,#2DD4BF,#5EEAD4)"
              : "linear-gradient(90deg,#475569,#64748B)",
          }}
        />
        <div
          className="absolute inset-y-0 w-[2px] bg-[#F0B94D]"
          style={{ left: `${threshold * 100}%` }}
          title={`Threshold ${threshold}`}
        />
      </div>
      <div className="flex justify-between mt-1 font-mono text-[11px] text-[#7C8AA5]">
        <span>0.0</span>
        <span className="text-[#F0B94D]">thr {threshold.toFixed(2)}</span>
        <span>1.0</span>
      </div>
    </div>
  );
}

function StatusPill({ ok, label }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono ${
        ok
          ? "bg-[#134E4A] text-[#5EEAD4]"
          : "bg-[#3A2318] text-[#F0B94D]"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-[#5EEAD4]" : "bg-[#F0B94D]"}`}
      />
      {label}
    </span>
  );
}

export default function DuplicateQuestionApp() {
  const [apiBase, setApiBase] = useState("http://localhost:8000");
  const [mode, setMode] = useState("search"); // "search" | "predict"

  // search state
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [searchResults, setSearchResults] = useState(null);

  // live/combined search state (local dataset + live Quora.com check)
  const [liveQuery, setLiveQuery] = useState("");
  const [combinedResults, setCombinedResults] = useState(null);

  // predict state
  const [q1, setQ1] = useState("");
  const [q2, setQ2] = useState("");
  const [predictResult, setPredictResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setSearchResults(null);
    try {
      const res = await fetch(`${apiBase}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: Number(topK) }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      setSearchResults(data);
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }, [apiBase, query, topK]);

  const runCombinedSearch = useCallback(async () => {
    if (!liveQuery.trim()) return;
    setLoading(true);
    setError(null);
    setCombinedResults(null);
    try {
      const res = await fetch(`${apiBase}/combined-search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: liveQuery, top_k: 5 }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      setCombinedResults(data);
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }, [apiBase, liveQuery]);

  const runPredict = useCallback(async () => {
    if (!q1.trim() || !q2.trim()) return;
    setLoading(true);
    setError(null);
    setPredictResult(null);
    try {
      const res = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question1: q1, question2: q2 }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      setPredictResult(data);
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }, [apiBase, q1, q2]);

  const handleSubmit = () => {
    if (mode === "search") return runSearch();
    if (mode === "live") return runCombinedSearch();
    return runPredict();
  };

  return (
    <div className="min-h-screen w-full bg-[#0B1220] text-[#E8ECF1] font-sans">
      <div className="max-w-3xl mx-auto px-6 py-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-2 text-[#5EEAD4] font-mono text-xs tracking-widest uppercase mb-1">
              <Zap size={13} />
              SBERT · FAISS retrieval
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Duplicate Question Scanner
            </h1>
          </div>
        </div>

        {/* API base config */}
        <div className="mb-6">
          <label className="block text-[11px] font-mono text-[#7C8AA5] mb-1.5 uppercase tracking-wide">
            API base URL
          </label>
          <input
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            className="w-full bg-[#121A2B] border border-[#1B2437] rounded-lg px-3 py-2 text-sm font-mono text-[#B8C2D6] focus:outline-none focus:ring-2 focus:ring-[#2DD4BF]/50 focus:border-[#2DD4BF]"
            placeholder="http://localhost:8000"
          />
        </div>

        {/* Mode tabs */}
        <div className="flex gap-2 mb-6">
          {[
            { key: "search", label: "Find similar", icon: Search },
            { key: "live", label: "Live Quora check", icon: Globe },
            { key: "predict", label: "Compare pair", icon: GitCompare },
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => {
                setMode(key);
                setError(null);
              }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                mode === key
                  ? "bg-[#132A2C] text-[#5EEAD4] border border-[#1F5C56]"
                  : "bg-[#121A2B] text-[#7C8AA5] border border-[#1B2437] hover:text-[#B8C2D6]"
              }`}
            >
              <Icon size={15} />
              {label}
            </button>
          ))}
        </div>

        {/* Input panel */}
        <div className="bg-[#121A2B] border border-[#1B2437] rounded-xl p-5 mb-6">
          {mode === "search" ? (
            <div className="space-y-3">
              <div>
                <label className="block text-[11px] font-mono text-[#7C8AA5] mb-1.5 uppercase tracking-wide">
                  Question
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={2}
                  placeholder="e.g. What is the step by step guide to invest in share market?"
                  className="w-full bg-[#0B1220] border border-[#1B2437] rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[#2DD4BF]/50 focus:border-[#2DD4BF]"
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <label className="text-[11px] font-mono text-[#7C8AA5] uppercase tracking-wide">
                    Top K
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={topK}
                    onChange={(e) => setTopK(e.target.value)}
                    className="w-16 bg-[#0B1220] border border-[#1B2437] rounded-md px-2 py-1 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-[#2DD4BF]/50"
                  />
                </div>
                <button
                  onClick={handleSubmit}
                  disabled={loading || !query.trim()}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#2DD4BF] text-[#0B1220] text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[#5EEAD4] transition-colors"
                >
                  {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
                  Scan
                </button>
              </div>
            </div>
          ) : mode === "live" ? (
            <div className="space-y-3">
              <div>
                <label className="block text-[11px] font-mono text-[#7C8AA5] mb-1.5 uppercase tracking-wide">
                  Question
                </label>
                <textarea
                  value={liveQuery}
                  onChange={(e) => setLiveQuery(e.target.value)}
                  rows={2}
                  placeholder="e.g. How do I use a specific tool released this year?"
                  className="w-full bg-[#0B1220] border border-[#1B2437] rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[#2DD4BF]/50 focus:border-[#2DD4BF]"
                />
              </div>
              <div className="text-xs text-[#7C8AA5]">
                Checks our local dataset (533K frozen questions) and a live Quora.com search side by side.
              </div>
              <div className="flex justify-end">
                <button
                  onClick={handleSubmit}
                  disabled={loading || !liveQuery.trim()}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#2DD4BF] text-[#0B1220] text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[#5EEAD4] transition-colors"
                >
                  {loading ? <Loader2 size={15} className="animate-spin" /> : <Globe size={15} />}
                  Check both
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <label className="block text-[11px] font-mono text-[#7C8AA5] mb-1.5 uppercase tracking-wide">
                  Question A
                </label>
                <textarea
                  value={q1}
                  onChange={(e) => setQ1(e.target.value)}
                  rows={2}
                  placeholder="How can I increase my height after 21?"
                  className="w-full bg-[#0B1220] border border-[#1B2437] rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[#2DD4BF]/50 focus:border-[#2DD4BF]"
                />
              </div>
              <div>
                <label className="block text-[11px] font-mono text-[#7C8AA5] mb-1.5 uppercase tracking-wide">
                  Question B
                </label>
                <textarea
                  value={q2}
                  onChange={(e) => setQ2(e.target.value)}
                  rows={2}
                  placeholder="Can height increase after 25?"
                  className="w-full bg-[#0B1220] border border-[#1B2437] rounded-lg px-3 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[#2DD4BF]/50 focus:border-[#2DD4BF]"
                />
              </div>
              <div className="flex justify-end">
                <button
                  onClick={handleSubmit}
                  disabled={loading || !q1.trim() || !q2.trim()}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#2DD4BF] text-[#0B1220] text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[#5EEAD4] transition-colors"
                >
                  {loading ? <Loader2 size={15} className="animate-spin" /> : <GitCompare size={15} />}
                  Compare
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 bg-[#3A2318] border border-[#5C3A1F] text-[#F0B94D] rounded-lg px-4 py-3 text-sm mb-6">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">Request failed</div>
              <div className="font-mono text-xs text-[#D9A96B] mt-0.5">{error}</div>
              <div className="text-xs text-[#D9A96B] mt-1">
                Check the API base URL and that the FastAPI server is running.
              </div>
            </div>
          </div>
        )}

        {/* Results: search */}
        {mode === "search" && searchResults && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-mono text-[#7C8AA5] uppercase tracking-wide">
                {searchResults.results.length} matches
              </span>
              <span className="text-xs font-mono text-[#7C8AA5]">
                {searchResults.latency_ms}ms
              </span>
            </div>
            {searchResults.results.length === 0 ? (
              <div className="text-sm text-[#7C8AA5] italic py-6 text-center border border-dashed border-[#1B2437] rounded-lg">
                No similar questions found in the index.
              </div>
            ) : (
              <div className="space-y-2.5">
                {searchResults.results.map((r, i) => (
                  <div
                    key={i}
                    className="bg-[#121A2B] border border-[#1B2437] rounded-lg px-4 py-3"
                  >
                    <div className="flex items-start justify-between gap-4 mb-2">
                      <span className="text-sm text-[#E8ECF1] leading-snug">{r.question}</span>
                      <span className="shrink-0 font-mono text-sm text-[#5EEAD4]">
                        {r.similarity.toFixed(3)}
                      </span>
                    </div>
                    <SimilarityBar value={r.similarity} />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Results: live/combined search */}
        {mode === "live" && combinedResults && (
          <div className="space-y-6">
            <div className="flex justify-end">
              <span className="text-xs font-mono text-[#7C8AA5]">
                {combinedResults.latency_ms}ms
              </span>
            </div>

            {/* Local dataset column */}
            <div>
              <div className="text-xs font-mono text-[#7C8AA5] uppercase tracking-wide mb-3">
                In our dataset ({combinedResults.local_results.length} matches)
              </div>
              {combinedResults.local_results.length === 0 ? (
                <div className="text-sm text-[#7C8AA5] italic py-4 text-center border border-dashed border-[#1B2437] rounded-lg">
                  Nothing close in the local dataset.
                </div>
              ) : (
                <div className="space-y-2.5">
                  {combinedResults.local_results.map((r, i) => (
                    <div key={i} className="bg-[#121A2B] border border-[#1B2437] rounded-lg px-4 py-3">
                      <div className="flex items-start justify-between gap-4 mb-2">
                        <span className="text-sm text-[#E8ECF1] leading-snug">{r.question}</span>
                        <span className="shrink-0 font-mono text-sm text-[#5EEAD4]">
                          {r.similarity.toFixed(3)}
                        </span>
                      </div>
                      <SimilarityBar value={r.similarity} />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Live Quora.com column */}
            <div>
              <div className="flex items-center gap-2 text-xs font-mono text-[#7C8AA5] uppercase tracking-wide mb-3">
                <Globe size={12} />
                Live on Quora.com ({combinedResults.live_results.length} matches)
              </div>
              {combinedResults.live_search_error ? (
                <div className="flex items-start gap-2 bg-[#3A2318] border border-[#5C3A1F] text-[#F0B94D] rounded-lg px-4 py-3 text-sm">
                  <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                  <div>
                    <div className="font-medium">Live check unavailable</div>
                    <div className="font-mono text-xs text-[#D9A96B] mt-0.5">
                      {combinedResults.live_search_error}
                    </div>
                  </div>
                </div>
              ) : combinedResults.live_results.length === 0 ? (
                <div className="text-sm text-[#7C8AA5] italic py-4 text-center border border-dashed border-[#1B2437] rounded-lg">
                  No matching question found live on Quora.
                </div>
              ) : (
                <div className="space-y-2.5">
                  {combinedResults.live_results.map((r, i) => (
                    <a
                      key={i}
                      href={r.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block bg-[#121A2B] border border-[#1B2437] rounded-lg px-4 py-3 hover:border-[#2DD4BF]/50 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3 mb-1">
                        <span className="text-sm text-[#E8ECF1] leading-snug">{r.title}</span>
                        <ExternalLink size={14} className="shrink-0 text-[#7C8AA5] mt-0.5" />
                      </div>
                      {r.snippet && (
                        <p className="text-xs text-[#7C8AA5] leading-snug line-clamp-2">
                          {r.snippet}
                        </p>
                      )}
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results: predict */}
        {mode === "predict" && predictResult && (
          <div className="bg-[#121A2B] border border-[#1B2437] rounded-lg px-5 py-5">
            <div className="flex items-center justify-between mb-4">
              <StatusPill
                ok={predictResult.is_duplicate}
                label={predictResult.is_duplicate ? "Duplicate" : "Not duplicate"}
              />
              <span className="text-xs font-mono text-[#7C8AA5]">
                {predictResult.latency_ms}ms
              </span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-[#7C8AA5] uppercase tracking-wide">
                Similarity
              </span>
              <span className="font-mono text-lg text-[#E8ECF1]">
                {predictResult.similarity.toFixed(4)}
              </span>
            </div>
            <SimilarityBar value={predictResult.similarity} threshold={predictResult.threshold} />
          </div>
        )}
      </div>
    </div>
  );
}