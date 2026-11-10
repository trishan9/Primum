"use client";

import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

const ROW = 56;
const GUTTER = 48;

interface RankColumnsProps {
  retrievers: string[];
  results: Record<string, Hit[]>;
  k: number;
  activeId: string | null;
  pinnedId: string | null;
  onHover: (id: string | null) => void;
  onPin: (id: string) => void;
}

function rankOf(hits: Hit[] | undefined, id: string): number | null {
  const index = hits?.findIndex((h) => h.passage.passage_id === id) ?? -1;
  return index === -1 ? null : index + 1;
}

/** How a passage moved relative to the ranking on its left, in words a screen reader can say. */
function movement(previous: number | null, current: number): { text: string; spoken: string } {
  if (previous === null) return { text: "new", spoken: "not in the ranking to the left" };
  const delta = previous - current;
  if (delta === 0) return { text: "=", spoken: "same rank as on the left" };
  return delta > 0
    ? { text: `+${delta}`, spoken: `up ${delta} from the left` }
    : { text: `−${-delta}`, spoken: `down ${-delta} from the left` };
}

/**
 * Rankings side by side, one column per retriever, with a line following each passage
 * from one ranking to the next. A steep line is a passage two retrievers disagree about;
 * a passage with no line into a column was not retrieved by it at all.
 */
export function RankColumns({ retrievers, results, k, activeId, pinnedId, onHover, onPin }: RankColumnsProps) {
  const template = retrievers.map(() => "minmax(0, 1fr)").join(` ${GUTTER}px `);
  const highlight = activeId ?? pinnedId;

  return (
    <div
      className="relative overflow-x-auto rounded-card border border-line bg-surface"
      tabIndex={0}
      role="group"
      aria-label="Rankings by retriever. Scroll sideways if the columns do not fit."
    >
      <div
        className="grid px-4 pb-4 pt-3"
        style={{ gridTemplateColumns: template, minWidth: retrievers.length * 184 + (retrievers.length - 1) * GUTTER + 32 }}
      >
        {retrievers.map((name, column) => {
          const hits = (results[name] ?? []).slice(0, k);
          const left = column > 0 ? results[retrievers[column - 1] as string] : undefined;
          return (
            <div key={name} className="contents">
              {column > 0 ? (
                <svg
                  aria-hidden="true"
                  width={GUTTER}
                  height={k * ROW}
                  viewBox={`0 0 ${GUTTER} ${k * ROW}`}
                  className="mt-9 overflow-visible"
                >
                  {(left ?? []).slice(0, k).map((hit, i) => {
                    const id = hit.passage.passage_id;
                    const to = rankOf(hits, id);
                    if (to === null) return null;
                    const y1 = (i + 0.5) * ROW;
                    const y2 = (to - 0.5) * ROW;
                    const on = highlight === id;
                    return (
                      <path
                        key={id}
                        d={`M0 ${y1} C ${GUTTER * 0.5} ${y1}, ${GUTTER * 0.5} ${y2}, ${GUTTER} ${y2}`}
                        fill="none"
                        stroke={on ? "var(--accent)" : "var(--axis)"}
                        strokeWidth={on ? 2.25 : 1}
                        opacity={highlight && !on ? 0.45 : 1}
                      />
                    );
                  })}
                </svg>
              ) : null}

              <section aria-labelledby={`rank-${column}`} className="min-w-0">
                <h3 id={`rank-${column}`} className="flex h-9 items-center justify-between gap-2 border-b border-line text-sm font-semibold text-ink">
                  <span className="truncate">{name}</span>
                  <span className="num shrink-0 text-xs font-normal text-muted">
                    {hits.length} of {k}
                  </span>
                </h3>
                {hits.length === 0 ? (
                  <p className="pt-3 text-sm text-muted">No passages returned.</p>
                ) : (
                  <ol>
                    {hits.map((hit, i) => {
                      const id = hit.passage.passage_id;
                      const on = highlight === id;
                      const move = column > 0 ? movement(rankOf(left, id), i + 1) : null;
                      return (
                        <li key={id} style={{ height: ROW }} className="border-b border-rule last:border-b-0">
                          <button
                            type="button"
                            onMouseEnter={() => onHover(id)}
                            onMouseLeave={() => onHover(null)}
                            onFocus={() => onHover(id)}
                            onBlur={() => onHover(null)}
                            onClick={() => onPin(id)}
                            aria-pressed={pinnedId === id}
                            className={`flex h-full w-full items-center gap-2.5 rounded-control px-1.5 text-left transition-colors duration-100 ${
                              on ? "bg-raised" : "hover:bg-ground"
                            }`}
                          >
                            <span className="num w-5 shrink-0 text-right text-sm font-semibold text-ink">{i + 1}</span>
                            <span className="min-w-0 flex-1">
                              <span className={`block truncate text-[0.8125rem] leading-tight text-ink ${on ? "font-semibold" : ""}`}>
                                {hit.passage.title || hit.passage.doc_id}
                              </span>
                              <span className="num mt-0.5 flex items-baseline gap-2 text-xs text-muted">
                                <span>score {fmtScore(hit.score)}</span>
                                {move ? (
                                  <span className={move.text === "new" ? "font-medium text-ink" : ""}>
                                    <span aria-hidden="true">{move.text}</span>
                                    <span className="sr-only">{move.spoken}</span>
                                  </span>
                                ) : null}
                              </span>
                            </span>
                          </button>
                        </li>
                      );
                    })}
                  </ol>
                )}
              </section>
            </div>
          );
        })}
      </div>
    </div>
  );
}
