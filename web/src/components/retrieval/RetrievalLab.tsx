"use client";

import { useId, useMemo, useRef, useState } from "react";

import { SOURCE_COPY } from "@/components/ask/copy";
import { ApiDown } from "@/components/shell/ApiDown";
import { useApi } from "@/components/shell/ApiProvider";
import { Button, Code, EmptyBlock, Panel } from "@/components/ui/controls";
import { ExternalGlyph } from "@/components/ui/glyphs";
import { ApiError, retrieve } from "@/lib/api";
import { fmtScore } from "@/lib/format";
import { SPEC_RETRIEVERS, type Hit, type RetrieveResponse } from "@/lib/types";

import { RankColumns } from "./RankColumns";

const DEFAULT_SELECTION = ["bm25", "dense-base", "dense-ft", "hybrid-ft"];
const K_OPTIONS = [5, 10, 20];
const CORPORA = [
  { id: "kb", label: "Knowledge base" },
  { id: "pubmedqa", label: "PubMedQA" },
  { id: "nfcorpus", label: "NFCorpus" },
];
/** What each retriever name means, from the system specification. Descriptions, not results. */
const RETRIEVER_NOTES: Record<string, string> = {
  bm25: "Okapi BM25 over words. Lexical: it matches the terms you typed, weighting rare ones more.",
  "dense-base": "A general-purpose sentence encoder. Matches meaning, with no biomedical training.",
  "dense-biomed": "An off-the-shelf encoder trained on PubMed text. The domain baseline.",
  "dense-ft": "This project's encoder: the base encoder fine-tuned contrastively on biomedical question and abstract pairs.",
  "hybrid-base": "BM25 and the base encoder, merged by reciprocal rank fusion.",
  "hybrid-ft": "BM25 and the fine-tuned encoder, merged by reciprocal rank fusion. The default for live answers.",
  "hybrid-ft+rerank": "The fused shortlist, reread and reordered by a cross-encoder.",
};

const EXAMPLES = ["statin muscle pain", "does metformin cause vitamin B12 deficiency", "il-6 inhibitors in covid-19"];

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; error: ApiError }
  | { kind: "ready"; query: string; k: number; retrievers: string[]; response: RetrieveResponse };

interface PassageLocation {
  hit: Hit;
  ranks: { retriever: string; rank: number | null; score: number | null }[];
}

export function RetrievalLab() {
  const id = useId();
  const { meta, reachability } = useApi();
  const available = meta?.retrievers?.length ? meta.retrievers : [...SPEC_RETRIEVERS];
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>(DEFAULT_SELECTION);
  const [k, setK] = useState(10);
  const [corpus, setCorpus] = useState("kb");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [problem, setProblem] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [pinned, setPinned] = useState<string | null>(null);
  const [phoneRetriever, setPhoneRetriever] = useState<string | null>(null);
  // Learned from the last successful comparison: false when the server has no concept dictionary,
  // in which case a `+concepts` retriever can only fail, so it is not offered.
  const [conceptsAvailable, setConceptsAvailable] = useState<boolean | null>(null);
  const controller = useRef<AbortController | null>(null);

  const needsConcepts = (name: string) => name.includes("+concepts");
  const offered = (name: string) => conceptsAvailable !== false || !needsConcepts(name);
  const ordered = available.filter((name) => selected.includes(name) && offered(name));

  const run = async () => {
    const text = query.trim();
    if (text.length < 2) return setProblem("Write a query of at least 2 characters.");
    if (ordered.length < 2) return setProblem("Choose at least two retrievers to compare.");
    setProblem(null);
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setStatus({ kind: "loading" });
    setPinned(null);
    setHovered(null);
    try {
      const response = await retrieve({ query: text, k, retrievers: ordered, corpus }, abort.signal);
      if (abort.signal.aborted) return;
      setStatus({ kind: "ready", query: text, k, retrievers: ordered, response });
      setConceptsAvailable(typeof response.concepts_available === "boolean" ? response.concepts_available : null);
      setPhoneRetriever(ordered[ordered.length - 1] ?? null);
    } catch (error) {
      if (abort.signal.aborted) return;
      setStatus({
        kind: "error",
        error: error instanceof ApiError ? error : new ApiError("internal", "The comparison failed for an unknown reason."),
      });
    }
  };

  const located = useMemo<PassageLocation | null>(() => {
    if (status.kind !== "ready" || !pinned) return null;
    let found: Hit | null = null;
    const ranks = status.retrievers.map((retriever) => {
      const hits = status.response.results[retriever] ?? [];
      const index = hits.findIndex((h) => h.passage.passage_id === pinned);
      const hit = index === -1 ? null : (hits[index] as Hit);
      if (hit && !found) found = hit;
      return { retriever, rank: hit ? index + 1 : null, score: hit ? hit.score : null };
    });
    return found ? { hit: found, ranks } : null;
  }, [status, pinned]);

  return (
    <div>
      <div className="max-w-[60rem]">
        <h1 className="font-serif text-[1.75rem] font-semibold leading-tight tracking-[-0.015em] text-ink">Retrieval lab</h1>
        <p className="mt-1 max-w-[68ch] text-[0.9375rem] leading-relaxed text-muted">
          One query, several retrievers, ranked side by side. A line follows each passage from one ranking to the
          next, so you can see what lexical search, dense search and their fusion agree on and where they part.
        </p>
      </div>

      <form
        className="mt-5 rounded-card border border-control bg-surface outline-offset-2 outline-accent focus-within:border-ink has-[input[type=search]:focus-visible]:outline-2"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
        noValidate
      >
        <label htmlFor={`${id}-query`} className="sr-only">
          Query
        </label>
        <input
          id={`${id}-query`}
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Type a query to compare rankings"
          aria-describedby={problem ? `${id}-problem` : undefined}
          className="block h-14 w-full rounded-t-card bg-transparent px-4 font-serif text-[1.25rem] text-ink placeholder:text-faint focus:outline-none sm:px-5"
        />

        <div className="flex flex-wrap items-end gap-x-6 gap-y-4 border-t border-rule px-4 py-3 sm:px-5">
          <fieldset className="min-w-0">
            <legend className="mb-1.5 text-[0.8125rem] font-medium text-ink">Retrievers, in column order</legend>
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {available.map((name) => (
                <label
                  key={name}
                  className={`flex min-h-11 items-center gap-2 text-sm sm:min-h-8 ${offered(name) ? "cursor-pointer text-ink" : "cursor-not-allowed text-muted"}`}
                  title={offered(name) ? undefined : "Needs the concept dictionary, which this server has not built"}
                >
                  <input
                    type="checkbox"
                    disabled={!offered(name)}
                    checked={selected.includes(name) && offered(name)}
                    onChange={(event) =>
                      setSelected((list) => (event.target.checked ? [...list, name] : list.filter((n) => n !== name)))
                    }
                    className="size-4 accent-[var(--primary)]"
                  />
                  {name}
                </label>
              ))}
            </div>
          </fieldset>

          <div>
            <label htmlFor={`${id}-k`} className="mb-1.5 block text-[0.8125rem] font-medium text-ink">
              Depth
            </label>
            <select
              id={`${id}-k`}
              value={k}
              onChange={(event) => setK(Number(event.target.value))}
              className="num h-11 rounded-control border border-control bg-surface px-2.5 text-sm text-ink sm:h-9"
            >
              {K_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  Top {option}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor={`${id}-corpus`} className="mb-1.5 block text-[0.8125rem] font-medium text-ink">
              Corpus
            </label>
            <select
              id={`${id}-corpus`}
              value={corpus}
              onChange={(event) => setCorpus(event.target.value)}
              className="h-11 max-w-full rounded-control border border-control bg-surface px-2.5 text-sm text-ink sm:h-9"
            >
              {CORPORA.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <Button type="submit" variant="primary" className="ml-auto min-w-28" disabled={status.kind === "loading"}>
            {status.kind === "loading" ? "Comparing" : "Compare"}
          </Button>
        </div>
      </form>
      {problem ? (
        <p id={`${id}-problem`} role="alert" className="mt-2 px-1 text-xs font-medium text-contradicted">
          {problem}
        </p>
      ) : null}

      <div className="mt-8" aria-busy={status.kind === "loading"}>
        {status.kind === "idle" ? (
          <>
            {reachability === "down" ? (
              <div className="mb-8">
                <ApiDown what="Rankings are computed by the server, so nothing can be compared until it is running." />
              </div>
            ) : null}
          <div className="grid gap-x-10 gap-y-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <div>
              <h2 className="text-[0.8125rem] font-medium text-muted">Queries to try</h2>
              <ul className="mt-1.5 divide-y divide-rule border-y border-rule">
                {EXAMPLES.map((example) => (
                  <li key={example}>
                    <button
                      type="button"
                      onClick={() => setQuery(example)}
                      className="flex min-h-11 w-full items-center rounded-control px-1 py-2 text-left font-serif text-[1.0625rem] leading-snug text-ink hover:bg-surface"
                    >
                      {example}
                    </button>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                Nothing is compared until you send a query: rankings are computed by the server, on its index.
              </p>
            </div>
            <div>
              <h2 className="text-[0.8125rem] font-medium text-muted">What the retrievers are</h2>
              <dl className="mt-1.5 divide-y divide-rule border-y border-rule">
                {available.map((name) => (
                  <div key={name} className="grid gap-x-4 gap-y-0.5 py-2.5 sm:grid-cols-[9.5rem_minmax(0,1fr)]">
                    <dt className="text-sm font-medium text-ink">{name}</dt>
                    <dd className="text-sm leading-relaxed text-muted">
                      {RETRIEVER_NOTES[name] ?? "Offered by the server; this page has no description for it."}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
          </>
        ) : null}

        {status.kind === "loading" ? (
          <EmptyBlock title={`Running ${ordered.length} retrievers`}>
            <p>The first query after a restart is slower, because the encoders load on first use.</p>
          </EmptyBlock>
        ) : null}

        {status.kind === "error" ? (
          status.error.kind === "unreachable" ? (
            <ApiDown what="The query was not sent." />
          ) : (
            <EmptyBlock
              title={
                status.error.kind === "no_index" || status.error.kind === "not_found"
                  ? "The index for this corpus has not been built"
                  : "The comparison failed"
              }
              action={<Button onClick={() => void run()}>Try again</Button>}
            >
              <p>{status.error.message}</p>
              {status.error.kind === "no_index" || status.error.kind === "not_found" ? (
                <p className="mt-2 flex flex-wrap gap-2">
                  <Code>primum data build {corpus}</Code>
                  <Code>primum index build {corpus}</Code>
                </p>
              ) : null}
            </EmptyBlock>
          )
        ) : null}

        {status.kind === "ready" ? (
          <div className="space-y-8">
            <QueryUnderstanding query={status.query} response={status.response} />

            <section aria-labelledby={`${id}-rankings`}>
              <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
                <h2 id={`${id}-rankings`} className="text-[0.9375rem] font-semibold text-ink">
                  Rankings for “{status.query}”
                </h2>
                <p className="hidden text-xs text-muted md:block">
                  Point at a passage to trace it across retrievers. Select it to read it. Movement is counted
                  against the column on the left.
                </p>
              </div>

              <div className="mt-3 hidden md:block">
                <RankColumns
                  retrievers={status.retrievers}
                  results={status.response.results}
                  k={status.k}
                  activeId={hovered}
                  pinnedId={pinned}
                  onHover={setHovered}
                  onPin={(passageId) => setPinned((current) => (current === passageId ? null : passageId))}
                />
              </div>

              <PhoneRanking
                retrievers={status.retrievers}
                results={status.response.results}
                k={status.k}
                current={phoneRetriever ?? status.retrievers[0] ?? ""}
                onChange={setPhoneRetriever}
                pinnedId={pinned}
                onPin={(passageId) => setPinned((current) => (current === passageId ? null : passageId))}
              />
            </section>

            {located ? (
              <Panel className="p-4 sm:p-5" aria-labelledby={`${id}-detail`}>
                <div className="flex flex-wrap items-baseline justify-between gap-3">
                  <p className="text-xs text-muted">
                    <span className="font-medium text-ink">
                      {SOURCE_COPY[located.hit.passage.source]?.label ?? located.hit.passage.source}
                    </span>
                    <span className="ml-2 font-mono">
                      {located.hit.passage.source === "pubmed" && /^\d+$/.test(located.hit.passage.doc_id) ? "PMID " : ""}
                      {located.hit.passage.doc_id}
                    </span>
                  </p>
                  <Button variant="quiet" onClick={() => setPinned(null)}>
                    Close
                  </Button>
                </div>
                <h2 id={`${id}-detail`} className="mt-1 font-serif text-[1.25rem] font-semibold leading-snug text-ink">
                  {located.hit.passage.title || "Untitled passage"}
                </h2>
                <div className="mt-3 grid gap-x-10 gap-y-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
                  <div>
                    <p className="max-w-[72ch] font-serif text-[1.0313rem] leading-[1.6] text-ink">{located.hit.passage.text}</p>
                    {located.hit.passage.url ? (
                      <a
                        href={located.hit.passage.url}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="mt-2 inline-flex min-h-11 items-center gap-1 text-sm text-link underline underline-offset-4 sm:min-h-0"
                      >
                        {SOURCE_COPY[located.hit.passage.source]?.open ?? "Open source"}
                        <ExternalGlyph size={13} />
                        <span className="sr-only">(opens in a new tab)</span>
                      </a>
                    ) : null}
                  </div>
                  <table className="num w-full self-start text-sm">
                    <caption className="pb-1 text-left text-xs text-muted">Where each retriever placed this passage</caption>
                    <thead>
                      <tr className="text-xs text-muted">
                        <th scope="col" className="py-1.5 text-left font-normal">
                          Retriever
                        </th>
                        <th scope="col" className="py-1.5 text-right font-normal">
                          Rank
                        </th>
                        <th scope="col" className="py-1.5 text-right font-normal">
                          Score
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {located.ranks.map((row) => (
                        <tr key={row.retriever} className="border-t border-rule">
                          <th scope="row" className="py-1.5 text-left font-normal text-ink">
                            {row.retriever}
                          </th>
                          <td className="py-1.5 text-right text-ink">
                            {row.rank ?? <span className="text-muted">not in top {status.k}</span>}
                          </td>
                          <td className="py-1.5 text-right text-ink">
                            {row.score === null ? <span className="text-muted">–</span> : fmtScore(row.score)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function QueryUnderstanding({ query, response }: { query: string; response: RetrieveResponse }) {
  const expanded = response.expanded_query ?? "";
  const added = expanded.startsWith(query) ? expanded.slice(query.length).trim() : expanded === query ? "" : null;

  return (
    <section aria-labelledby="understanding-heading" className="grid gap-x-10 gap-y-6 lg:grid-cols-2">
      <h2 id="understanding-heading" className="sr-only">
        How the query was read
      </h2>
      <div>
        <h3 className="text-[0.9375rem] font-semibold text-ink">Concepts recognised</h3>
        {response.concepts_available === false ? (
          <p className="mt-1.5 text-sm text-muted">
            Not available: this server has no concept dictionary (it is built with the <code>kb</code> corpus), so no
            term was linked and retrievers that expand queries with concepts are not offered.
          </p>
        ) : response.concepts.length === 0 ? (
          <p className="mt-1.5 text-sm text-muted">No term in the query matched the concept dictionary.</p>
        ) : (
          <table className="mt-1.5 w-full text-sm">
            <thead>
              <tr className="text-xs text-muted">
                <th scope="col" className="py-1.5 pr-3 text-left font-normal">
                  In the query
                </th>
                <th scope="col" className="py-1.5 pr-3 text-left font-normal">
                  Preferred term
                </th>
                <th scope="col" className="py-1.5 text-left font-normal">
                  MeSH
                </th>
              </tr>
            </thead>
            <tbody>
              {response.concepts.map((concept, i) => (
                <tr key={`${concept.preferred}-${i}`} className="border-t border-rule align-baseline">
                  <td className="py-1.5 pr-3 font-serif text-[1rem] italic text-ink">{concept.surface}</td>
                  <td className="py-1.5 pr-3 text-ink">
                    {concept.url ? (
                      <a href={concept.url} target="_blank" rel="noreferrer noopener" className="text-link underline underline-offset-4">
                        {concept.preferred}
                        <span className="sr-only"> (opens MedlinePlus in a new tab)</span>
                      </a>
                    ) : (
                      concept.preferred
                    )}
                    {concept.synonyms.length > 0 ? (
                      <span className="block text-xs text-muted">also: {concept.synonyms.slice(0, 4).join(", ")}</span>
                    ) : null}
                  </td>
                  <td className="py-1.5 font-mono text-xs text-muted">{concept.mesh_ids.join(", ") || "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div>
        <h3 className="text-[0.9375rem] font-semibold text-ink">Expanded query</h3>
        {added === "" ? (
          <p className="mt-1.5 text-sm text-muted">Nothing was added: the query is searched as typed.</p>
        ) : (
          <>
            <p className="mt-1.5 font-serif text-[1.0625rem] leading-relaxed text-ink">
              {added === null ? (
                expanded
              ) : (
                <>
                  {query} <mark>{added}</mark>
                </>
              )}
            </p>
            <p className="mt-1 text-xs text-muted">
              Synonyms of the recognised concepts are appended, so lexical search can match documents that use a
              different name for the same thing.
            </p>
          </>
        )}
      </div>
    </section>
  );
}

interface PhoneRankingProps {
  retrievers: string[];
  results: Record<string, Hit[]>;
  k: number;
  current: string;
  onChange: (retriever: string) => void;
  pinnedId: string | null;
  onPin: (id: string) => void;
}

/** On a narrow screen one ranking shows at a time, and each row says where the others put that passage. */
function PhoneRanking({ retrievers, results, k, current, onChange, pinnedId, onPin }: PhoneRankingProps) {
  const id = useId();
  const hits = (results[current] ?? []).slice(0, k);
  const others = retrievers.filter((name) => name !== current);

  return (
    <div className="mt-3 md:hidden">
      <label htmlFor={`${id}-retriever`} className="mb-1.5 block text-[0.8125rem] font-medium text-ink">
        Ranking by
      </label>
      <select
        id={`${id}-retriever`}
        value={current}
        onChange={(event) => onChange(event.target.value)}
        className="h-11 w-full rounded-control border border-control bg-surface px-2.5 text-sm text-ink"
      >
        {retrievers.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>

      {hits.length === 0 ? (
        <p className="mt-3 text-sm text-muted">No passages returned.</p>
      ) : (
        <ol className="mt-3 divide-y divide-rule border-y border-rule">
          {hits.map((hit, i) => {
            const passageId = hit.passage.passage_id;
            return (
              <li key={passageId}>
                <button
                  type="button"
                  onClick={() => onPin(passageId)}
                  aria-pressed={pinnedId === passageId}
                  className={`flex w-full gap-3 px-1 py-3 text-left ${pinnedId === passageId ? "bg-raised" : ""}`}
                >
                  <span className="num w-5 shrink-0 text-right text-sm font-semibold text-ink">{i + 1}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm leading-snug text-ink">{hit.passage.title || hit.passage.doc_id}</span>
                    <span className="num mt-1 block text-xs leading-relaxed text-muted">
                      score {fmtScore(hit.score)}
                      {others.map((name) => {
                        const index = (results[name] ?? []).slice(0, k).findIndex((h) => h.passage.passage_id === passageId);
                        return (
                          <span key={name} className="whitespace-nowrap">
                            {", "}
                            {name} {index === -1 ? "none" : `#${index + 1}`}
                          </span>
                        );
                      })}
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
