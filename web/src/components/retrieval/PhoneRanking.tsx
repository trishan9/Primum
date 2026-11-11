"use client";

import { SOURCE_COPY } from "@/components/ask/copy";
import { cn } from "@/components/ui/cn";
import { Tab, TabPanel, Tabs, TabsList } from "@/components/ui/tabs";
import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

import { passageHeading, rankOf, scoreKind } from "./retrievers";

interface PhoneRankingProps {
  retrievers: string[];
  results: Record<string, Hit[]>;
  k: number;
  current: string;
  onChange: (retriever: string) => void;
  pinnedId: string | null;
  onPin: (id: string) => void;
}

/**
 * On a narrow screen one ranking shows at a time, as tabs, and each row says where the other
 * retrievers put the same passage: the lines of the wide view, said in words.
 */
export function PhoneRanking({ retrievers, results, k, current, onChange, pinnedId, onPin }: PhoneRankingProps) {
  return (
    <Tabs value={current} onValueChange={onChange}>
      <TabsList label="Ranking by" className="px-2">
        {retrievers.map((name) => (
          <Tab key={name} value={name} className="font-mono text-[0.8125rem]">
            {name}
          </Tab>
        ))}
      </TabsList>
      {retrievers.map((name) => {
        const hits = (results[name] ?? []).slice(0, k);
        const others = retrievers.filter((other) => other !== name);
        return (
          <TabPanel key={name} value={name} className="pt-1">
            {hits.length === 0 ? (
              <p className="px-4 py-4 text-sm text-muted">No passages returned.</p>
            ) : (
              <ol className="divide-y divide-rule">
                {hits.map((hit, i) => {
                  const id = hit.passage.passage_id;
                  const pinned = pinnedId === id;
                  return (
                    <li key={id}>
                      <button
                        type="button"
                        onClick={() => onPin(id)}
                        aria-pressed={pinned}
                        className={cn(
                          "flex w-full gap-3 px-4 py-3 text-left transition-colors duration-[var(--dur-fast)]",
                          pinned ? "bg-accent-soft" : "active:bg-raised",
                        )}
                      >
                        <span className={cn("num w-5 shrink-0 text-right text-sm font-semibold", pinned ? "text-accent-ink" : "text-ink")}>
                          {i + 1}
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className={cn("line-clamp-2 text-sm leading-snug", pinned ? "font-medium text-accent-ink" : "text-ink")}>
                            {passageHeading(hit.passage)}
                          </span>
                          <span className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
                            <span>{SOURCE_COPY[hit.passage.source]?.label ?? hit.passage.source}</span>
                            <span className="num rounded-chip bg-raised px-1.5 py-0.5 text-ink">
                              <span className="text-muted">{scoreKind(name)}</span> {fmtScore(hit.score)}
                            </span>
                          </span>
                          <span className="mt-1.5 flex flex-wrap gap-1">
                            {others.map((other) => {
                              const rank = rankOf((results[other] ?? []).slice(0, k), id);
                              return (
                                <span
                                  key={other}
                                  className={cn(
                                    "num inline-flex items-center gap-1 rounded-chip border px-1.5 py-0.5 text-xs",
                                    rank === null ? "border-dashed border-frame text-faint" : "border-line text-muted",
                                  )}
                                >
                                  <span className="font-mono">{other}</span>
                                  <span className={rank === null ? "" : "font-medium text-ink"}>
                                    {rank === null ? (
                                      <>
                                        <span aria-hidden="true">–</span>
                                        <span className="sr-only">not in top {k}</span>
                                      </>
                                    ) : (
                                      `#${rank}`
                                    )}
                                  </span>
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
          </TabPanel>
        );
      })}
    </Tabs>
  );
}
