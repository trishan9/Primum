"use client";

import { useCallback, useId, useMemo, useRef, useState } from "react";

import { ApiDown } from "@/components/shell/ApiDown";
import { useApi } from "@/components/shell/ApiProvider";
import { StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Code } from "@/components/ui/code";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoGlyph, WarningGlyph } from "@/components/ui/glyphs";
import { ApiError, retrieve } from "@/lib/api";
import { bumpTakeaway, rankBump } from "@/lib/charts/retrieval";
import { scrollBehavior } from "@/lib/motion";
import { SPEC_RETRIEVERS, type Hit, type RetrieveResponse } from "@/lib/types";

import { IdleState } from "./IdleState";
import { PassageDetail, type PassagePlacement } from "./PassageDetail";
import { PhoneRanking } from "./PhoneRanking";
import { QueryBar, RetrieverChips } from "./QueryBar";
import { AgreementStats, QueryReading } from "./QueryReading";
import { RankColumns, RankColumnsSkeleton } from "./RankColumns";
import { OverlapCard, ScoreProfilesCard } from "./RetrievalFigures";
import { CORPORA, DEFAULT_SELECTION, K_OPTIONS, agreement, needsConcepts } from "./retrievers";

type Status =
  | { kind: "idle" }
  | { kind: "loading"; retrievers: string[]; k: number }
  | { kind: "error"; error: ApiError; corpus: string }
  | { kind: "ready"; query: string; k: number; corpus: string; retrievers: string[]; response: RetrieveResponse };

const numberFormat = new Intl.NumberFormat("en");

/** The playground: one query, several retrievers, rankings side by side. State lives here; the parts are presentational. */
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
  const detailRef = useRef<HTMLDivElement>(null);

  const offered = (name: string) => conceptsAvailable !== false || !needsConcepts(name);
  const ordered = available.filter((name) => selected.includes(name) && offered(name));

  const run = async (text = query.trim()) => {
    if (text.length < 2) return setProblem("Write a query of at least 2 characters.");
    if (ordered.length < 2) return setProblem("Choose at least two retrievers to compare.");
    setProblem(null);
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setStatus({ kind: "loading", retrievers: ordered, k });
    setPinned(null);
    setHovered(null);
    try {
      const response = await retrieve({ query: text, k, retrievers: ordered, corpus }, abort.signal);
      if (abort.signal.aborted) return;
      setStatus({ kind: "ready", query: text, k, corpus, retrievers: ordered, response });
      setConceptsAvailable(typeof response.concepts_available === "boolean" ? response.concepts_available : null);
      setPhoneRetriever(ordered[0] ?? null);
    } catch (error) {
      if (abort.signal.aborted) return;
      setStatus({
        kind: "error",
        corpus,
        error: error instanceof ApiError ? error : new ApiError("internal", "The comparison failed for an unknown reason."),
      });
    }
  };

  // Stable, so memoised cards do not re-render on every hover. The detail panel exists only while
  // a passage is pinned, so the scroll is a no-op after an unpin.
  const pin = useCallback((passageId: string) => {
    setPinned((current) => (current === passageId ? null : passageId));
    requestAnimationFrame(() => detailRef.current?.scrollIntoView({ block: "nearest", behavior: scrollBehavior() }));
  }, []);
  const unpin = useCallback(() => setPinned(null), []);

  // Derived from the response only, so hovering never recomputes them.
  const agreementStats = useMemo(
    () => (status.kind === "ready" ? agreement(status.retrievers, status.response.results, status.k) : null),
    [status],
  );
  const spread = useMemo(() => {
    if (status.kind !== "ready") return null;
    const columns = status.retrievers.filter((r) => Array.isArray(status.response.results[r]));
    return bumpTakeaway(rankBump(status.response.results, status.retrievers, status.k), columns.length, status.k);
  }, [status]);

  const located = useMemo<{ hit: Hit; placements: PassagePlacement[] } | null>(() => {
    if (status.kind !== "ready" || !pinned) return null;
    let found: Hit | null = null;
    const placements = status.retrievers.map((retriever) => {
      const hits = (status.response.results[retriever] ?? []).slice(0, status.k);
      const index = hits.findIndex((h) => h.passage.passage_id === pinned);
      const hit = index === -1 ? null : (hits[index] as Hit);
      if (hit && !found) found = hit;
      return { retriever, rank: hit ? index + 1 : null, score: hit ? hit.score : null };
    });
    return found ? { hit: found, placements } : null;
  }, [status, pinned]);

  const unavailableConcepts = conceptsAvailable === false && available.some(needsConcepts);
  const corpusSize = (corpusId: string) => meta?.corpora?.[corpusId];

  return (
    <div className="flex flex-col gap-6">
      <QueryBar onSubmit={() => void run()}>
        <QueryBar.Input
          id={`${id}-query`}
          value={query}
          onChange={(value) => {
            setQuery(value);
            if (problem) setProblem(null);
          }}
          busy={status.kind === "loading"}
          describedBy={problem ? `${id}-problem` : undefined}
        />
        <QueryBar.Options>
          <RetrieverChips
            available={available}
            selected={selected}
            isOffered={offered}
            onToggle={(name) =>
              setSelected((list) => (list.includes(name) ? list.filter((n) => n !== name) : [...list, name]))
            }
          />
          <QueryBar.Settings>
            <QueryBar.Select id={`${id}-k`} label="Depth" value={k} onChange={(value) => setK(Number(value))} className="w-28">
              {K_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  Top {option}
                </option>
              ))}
            </QueryBar.Select>
            <QueryBar.Select id={`${id}-corpus`} label="Corpus" value={corpus} onChange={setCorpus} className="min-w-0 basis-full sm:w-72 sm:basis-auto">
              {CORPORA.map((option) => {
                const size = corpusSize(option.id);
                return (
                  <option key={option.id} value={option.id}>
                    {option.label}
                    {typeof size === "number" ? ` (${numberFormat.format(size)} passages)` : size === null ? " (not built)" : ""}
                  </option>
                );
              })}
            </QueryBar.Select>
          </QueryBar.Settings>
        </QueryBar.Options>
        {problem ? (
          <QueryBar.Note tone="problem" id={`${id}-problem`}>
            <WarningGlyph size={14} className="mt-0.5" />
            {problem}
          </QueryBar.Note>
        ) : unavailableConcepts ? (
          <QueryBar.Note>
            <InfoGlyph size={14} className="mt-0.5 text-faint" />
            <span>
              Concept retrievers are switched off: this server has no concept dictionary. Build it with{" "}
              <Code>primum data build kb</Code>.
            </span>
          </QueryBar.Note>
        ) : null}
      </QueryBar>

      <p role="status" className="sr-only">
        {status.kind === "loading"
          ? `Ranking with ${status.retrievers.length} retrievers`
          : status.kind === "ready"
            ? `Rankings ready for ${status.query}`
            : status.kind === "error"
              ? "The comparison failed"
              : ""}
      </p>
      <div aria-busy={status.kind === "loading"} className="flex flex-col gap-6">
        {status.kind === "idle" ? (
          <>
            {reachability === "down" ? (
              <ApiDown what="Rankings are computed by the server, so nothing can be compared until it is running." />
            ) : null}
            <IdleState
              available={available}
              onTry={(example) => {
                setQuery(example);
                void run(example);
              }}
            />
          </>
        ) : null}

        {status.kind === "loading" ? (
          <Card>
            <Card.Header>
              <Card.Heading>
                <Card.Title className="flex items-center gap-2">
                  <StatusDot tone="busy" />
                  Ranking with {status.retrievers.length} retrievers
                </Card.Title>
                <Card.Description>The first query after a restart is slower, because the encoders load on first use.</Card.Description>
              </Card.Heading>
            </Card.Header>
            <Card.Body flush className="pt-3">
              <RankColumnsSkeleton retrievers={status.retrievers} k={status.k} />
            </Card.Body>
          </Card>
        ) : null}

        {status.kind === "error" ? <ErrorState status={status} onRetry={() => void run()} /> : null}

        {status.kind === "ready" ? (
          <>
            <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
              <QueryReading query={status.query} response={status.response} />
              <div className="grid min-w-0 gap-4">
                {agreementStats ? <AgreementStats agreement={agreementStats} k={status.k} summary={spread ?? undefined} /> : null}
                <OverlapCard retrievers={status.retrievers} results={status.response.results} k={status.k} />
              </div>
            </div>

            <Card aria-labelledby={`${id}-rankings`}>
              <Card.Header>
                <Card.Heading>
                  <Card.Title id={`${id}-rankings`}>Rankings for “{status.query}”</Card.Title>
                  <Card.Description>
                    Top {status.k} from {CORPORA.find((c) => c.id === status.corpus)?.label ?? status.corpus}.{" "}
                    <span className="hidden md:inline">
                      Point at a passage to trace it; select it to read it. Movement is counted against the column on the left.
                    </span>
                    <span className="md:hidden">Select a passage to read it.</span>
                  </Card.Description>
                </Card.Heading>
                <Card.Actions className="hidden md:flex">
                  <Legend />
                </Card.Actions>
              </Card.Header>
              <Card.Body flush className="pt-2">
                <div className="hidden md:block">
                  <RankColumns
                    retrievers={status.retrievers}
                    results={status.response.results}
                    k={status.k}
                    activeId={hovered}
                    pinnedId={pinned}
                    onHover={setHovered}
                    onPin={pin}
                  />
                </div>
                <div className="md:hidden">
                  <PhoneRanking
                    retrievers={status.retrievers}
                    results={status.response.results}
                    k={status.k}
                    current={phoneRetriever ?? status.retrievers[0] ?? ""}
                    onChange={setPhoneRetriever}
                    pinnedId={pinned}
                    onPin={pin}
                  />
                </div>
              </Card.Body>
            </Card>

            {located ? (
              <PassageDetail
                ref={detailRef}
                hit={located.hit}
                placements={located.placements}
                k={status.k}
                onClose={unpin}
              />
            ) : (
              <p className="flex items-center gap-2 px-1 text-meta text-muted">
                <InfoGlyph size={14} className="text-faint" />
                Select any passage to read it in full and see where every retriever placed it.
              </p>
            )}

            <ScoreProfilesCard
              retrievers={status.retrievers}
              results={status.response.results}
              k={status.k}
              activeId={hovered}
              pinnedId={pinned}
              onHover={setHovered}
              onPin={pin}
            />
          </>
        ) : null}
      </div>
    </div>
  );
}

/** What the lines and badges in the columns mean. */
function Legend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted" aria-label="Legend">
      <li className="flex items-center gap-1.5">
        <svg aria-hidden="true" width="22" height="10" viewBox="0 0 22 10" fill="none" stroke="var(--axis)">
          <path d="M0 2 C 11 2, 11 8, 22 8" />
        </svg>
        Same passage
      </li>
      <li className="flex items-center gap-1.5">
        <svg aria-hidden="true" width="22" height="10" viewBox="0 0 22 10" fill="none" stroke="var(--axis)">
          <path d="M0 5 H12" />
          <circle cx="15" cy="5" r="2.5" />
        </svg>
        Dropped by the next
      </li>
      <li className="flex items-center gap-1.5">
        <span className="rounded-chip bg-accent-soft px-1 text-accent-ink">new</span>
        Not in the column to the left
      </li>
    </ul>
  );
}

function ErrorState({ status, onRetry }: { status: Extract<Status, { kind: "error" }>; onRetry: () => void }) {
  const { error, corpus } = status;
  if (error.kind === "unreachable") return <ApiDown what="The query was not sent." />;
  const noIndex = error.kind === "no_index" || error.kind === "not_found";
  return (
    <EmptyState>
      <EmptyState.Main>
        <EmptyState.Title>{noIndex ? "The index for this corpus has not been built" : "The comparison failed"}</EmptyState.Title>
        <EmptyState.Description>
          <p>{error.message}</p>
          {noIndex ? (
            <p className="mt-2 flex flex-wrap gap-2">
              <Code>primum data build {corpus}</Code>
              <Code>primum index build {corpus}</Code>
            </p>
          ) : null}
        </EmptyState.Description>
        <EmptyState.Actions>
          <Button onClick={onRetry}>Try again</Button>
        </EmptyState.Actions>
      </EmptyState.Main>
    </EmptyState>
  );
}
