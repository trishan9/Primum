"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/components/ui/cn";
import { SOURCE_COPY } from "@/components/ask/copy";
import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

import { docLabel, familyOf, movement, passageHeading, rankOf, scoreKind, type Movement } from "./retrievers";

const ROW = 60;
const GUTTER = 56;
const HEAD = 52;
const COLUMN_MIN = 208;
/** Length of the stub drawn for a passage that leaves or enters between two rankings. */
const STUB = 12;

interface RankColumnsProps {
  retrievers: string[];
  results: Record<string, Hit[]>;
  k: number;
  activeId: string | null;
  pinnedId: string | null;
  onHover: (id: string | null) => void;
  onPin: (id: string) => void;
}

/** A grid template of columns with a gutter between each pair, where the lines are drawn. */
function template(count: number): { gridTemplateColumns: string; minWidth: number } {
  return {
    gridTemplateColumns: Array.from({ length: count }, () => `minmax(${COLUMN_MIN}px, 1fr)`).join(` ${GUTTER}px `),
    minWidth: count * COLUMN_MIN + (count - 1) * GUTTER,
  };
}

/**
 * Rankings side by side, one column per retriever, with a line following each passage from one
 * ranking to the next. A steep line is a passage two retrievers disagree about; a stub that ends in
 * a ring is a passage the next ranking left out, and one that starts from a ring is new to it.
 */
export function RankColumns({ retrievers, results, k, activeId, pinnedId, onHover, onPin }: RankColumnsProps) {
  const highlight = activeId ?? pinnedId;

  return (
    <div
      className="overflow-x-auto overscroll-x-contain focus-visible:outline-offset-[-2px]"
      tabIndex={0}
      role="group"
      aria-label="Rankings by retriever. Scroll sideways if the columns do not fit."
    >
      <div className="grid px-4 pb-4 sm:px-5" style={template(retrievers.length)}>
        {retrievers.map((name, column) => {
          const hits = (results[name] ?? []).slice(0, k);
          const left = column > 0 ? (results[retrievers[column - 1] as string] ?? []).slice(0, k) : undefined;
          return (
            <div key={name} className="contents">
              {left ? <Links left={left} right={hits} k={k} highlight={highlight} /> : null}
              <section aria-labelledby={`rank-${column}`} className="min-w-0">
                <ColumnHead id={`rank-${column}`} name={name} count={hits.length} k={k} />
                {hits.length === 0 ? (
                  <p className="pt-3 text-sm text-muted">No passages returned.</p>
                ) : (
                  <ol>
                    {hits.map((hit, i) => {
                      const id = hit.passage.passage_id;
                      return (
                        <RankRow
                          key={id}
                          rank={i + 1}
                          hit={hit}
                          kind={scoreKind(name)}
                          move={left ? movement(rankOf(left, id), i + 1) : null}
                          lit={highlight === id}
                          dim={highlight !== null && highlight !== id}
                          pinned={pinnedId === id}
                          onHover={onHover}
                          onPin={onPin}
                        />
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

function ColumnHead({ id, name, count, k }: { id: string; name: string; count: number; k: number }) {
  return (
    <h3 id={id} className="flex items-center justify-between gap-2 border-b border-line" style={{ height: HEAD }}>
      <span className="flex min-w-0 flex-col">
        <span className="truncate font-mono text-[0.8125rem] font-medium text-ink">{name}</span>
        <span className="text-xs font-normal text-muted">{familyOf(name)}</span>
      </span>
      <span className="num shrink-0 text-xs font-normal text-muted">
        {count} of {k}
      </span>
    </h3>
  );
}

function RankRow({
  rank,
  hit,
  kind,
  move,
  lit,
  dim,
  pinned,
  onHover,
  onPin,
}: {
  rank: number;
  hit: Hit;
  kind: string;
  move: Movement | null;
  lit: boolean;
  dim: boolean;
  pinned: boolean;
  onHover: (id: string | null) => void;
  onPin: (id: string) => void;
}) {
  const id = hit.passage.passage_id;
  const source = SOURCE_COPY[hit.passage.source]?.label ?? hit.passage.source;
  return (
    <li style={{ height: ROW }} className="py-1">
      <button
        type="button"
        onMouseEnter={() => onHover(id)}
        onMouseLeave={() => onHover(null)}
        onFocus={() => onHover(id)}
        onBlur={() => onHover(null)}
        onClick={() => onPin(id)}
        aria-pressed={pinned}
        title={`${passageHeading(hit.passage)} (${docLabel(hit.passage.source, hit.passage.doc_id)})`}
        className={cn(
          "relative flex h-full w-full items-center gap-2.5 rounded-control px-2 text-left transition-[background-color,opacity] duration-[var(--dur-fast)]",
          pinned ? "bg-accent-soft" : lit ? "bg-raised" : "hover:bg-raised",
          dim && !pinned ? "opacity-60" : "",
        )}
      >
        {lit || pinned ? <span aria-hidden="true" className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-accent" /> : null}
        <span className={cn("num w-5 shrink-0 text-right text-sm font-semibold", pinned ? "text-accent-ink" : "text-ink")}>
          {rank}
        </span>
        <span className="min-w-0 flex-1">
          <span className={cn("block truncate text-[0.8125rem] leading-[1.125rem]", pinned ? "font-medium text-accent-ink" : "text-ink")}>
            {passageHeading(hit.passage)}
          </span>
          <span className="mt-1 flex items-center gap-1.5 text-xs leading-none text-muted">
            <span className="truncate">{source}</span>
            <span className="num ml-auto inline-flex shrink-0 items-center gap-1 rounded-chip bg-raised px-1.5 py-[3px] text-ink">
              <span className="text-muted">{kind}</span>
              {fmtScore(hit.score)}
            </span>
            {move ? <MoveBadge move={move} /> : null}
          </span>
        </span>
      </button>
    </li>
  );
}

function MoveBadge({ move }: { move: Movement }) {
  return (
    <span
      className={cn(
        "num inline-flex min-w-7 shrink-0 items-center justify-center rounded-chip px-1 py-[3px]",
        move.direction === "new" ? "bg-accent-soft font-medium text-accent-ink" : "border border-line text-muted",
      )}
    >
      <span aria-hidden="true">{move.text}</span>
      <span className="sr-only">{move.spoken}</span>
    </span>
  );
}

/** The lines between two neighbouring rankings. Drawn, not read: the rows carry the same facts in words. */
function Links({ left, right, k, highlight }: { left: Hit[]; right: Hit[]; k: number; highlight: string | null }) {
  const height = k * ROW;
  const y = (rank: number) => (rank - 0.5) * ROW;
  const line = (id: string) => ({
    stroke: highlight === id ? "var(--accent)" : "var(--axis)",
    strokeWidth: highlight === id ? 2.25 : 1,
    opacity: highlight && highlight !== id ? 0.3 : 1,
  });

  return (
    <svg
      aria-hidden="true"
      width={GUTTER}
      height={height}
      viewBox={`0 0 ${GUTTER} ${height}`}
      className="overflow-visible"
      style={{ marginTop: HEAD }}
      fill="none"
    >
      {left.map((hit, i) => {
        const id = hit.passage.passage_id;
        const to = rankOf(right, id);
        const y1 = y(i + 1);
        if (to === null) {
          return (
            <g key={id} {...line(id)}>
              <path d={`M0 ${y1} H${STUB}`} />
              <circle cx={STUB + 2.5} cy={y1} r={2.5} />
            </g>
          );
        }
        const y2 = y(to);
        return <path key={id} d={`M0 ${y1} C ${GUTTER * 0.5} ${y1}, ${GUTTER * 0.5} ${y2}, ${GUTTER} ${y2}`} {...line(id)} />;
      })}
      {right.map((hit, i) => {
        const id = hit.passage.passage_id;
        if (rankOf(left, id) !== null) return null;
        const y2 = y(i + 1);
        return (
          <g key={id} {...line(id)}>
            <circle cx={GUTTER - STUB - 2.5} cy={y2} r={2.5} />
            <path d={`M${GUTTER - STUB} ${y2} H${GUTTER}`} />
          </g>
        );
      })}
    </svg>
  );
}

/** The same frame while the server ranks: one column per retriever, rows at their real height. */
export function RankColumnsSkeleton({ retrievers, k }: { retrievers: string[]; k: number }) {
  return (
    <div className="overflow-hidden" aria-hidden="true">
      <div className="flex flex-col px-4 pb-2 md:hidden">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className="flex items-start gap-3 border-b border-rule py-3 last:border-b-0">
            <Skeleton className="mt-0.5 h-3.5 w-4" />
            <span className="flex flex-1 flex-col gap-2">
              <Skeleton className={i % 2 ? "h-3.5 w-4/5" : "h-3.5 w-full"} />
              <Skeleton className="h-3 w-2/5" />
            </span>
          </div>
        ))}
      </div>
      <div className="hidden px-4 pb-4 sm:px-5 md:grid" style={template(retrievers.length)}>
        {retrievers.map((name, column) => (
          <div key={name} className="contents">
            {column > 0 ? <div /> : null}
            <div className="min-w-0">
              <div className="flex flex-col justify-center border-b border-line" style={{ height: HEAD }}>
                <span className="truncate font-mono text-[0.8125rem] font-medium text-ink">{name}</span>
                <span className="text-xs text-muted">{familyOf(name)}</span>
              </div>
              {Array.from({ length: Math.min(k, 6) }, (_, i) => (
                <div key={i} className="flex items-center gap-2.5 px-2" style={{ height: ROW }}>
                  <Skeleton className="h-3.5 w-4" />
                  <span className="flex flex-1 flex-col gap-1.5">
                    <Skeleton className={i % 2 ? "h-3 w-4/5" : "h-3 w-full"} />
                    <Skeleton className="h-2.5 w-1/2" />
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
