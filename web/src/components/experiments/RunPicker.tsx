"use client";

import type { ReactNode } from "react";

import { fmtInt } from "@/lib/format";
import { runLabel } from "@/lib/results";
import type { RunInfo } from "@/lib/types";

import { Disclosure } from "@/components/ui/disclosure";
import { Picker } from "@/components/ui/picker";

/** "N failed generations (rag), excluded from scoring", from the run's own counts. */
function failedText(info: RunInfo): string | null {
  if (!info.n_failed) return null;
  const entries = Object.entries(info.n_failed).filter(([, n]) => typeof n === "number");
  const total = entries.reduce((sum, [, n]) => sum + n, 0);
  if (entries.length === 0) return null;
  if (total === 0) return "no failed generations";
  const failing = entries.filter(([, n]) => n > 0);
  const where = failing.length === 1 ? failing[0]?.[0] : failing.map(([c, n]) => `${fmtInt(n)} in ${c}`).join(", ");
  return `${fmtInt(total)} failed generation${total === 1 ? "" : "s"} (${where}), excluded from scoring`;
}

/** The run switch, for the page header's action slot. */
export function RunSwitch({ runs, run, onChange }: { runs: RunInfo[]; run: string; onChange: (run: string) => void }) {
  return (
    <Picker
      legend="Generator whose answers RQ2 and RQ3 score"
      value={run}
      options={runs.map((r) => ({ value: r.run, label: runLabel(r, runs) }))}
      onChange={onChange}
      segmentedFrom="lg"
      className="w-full lg:w-auto"
    />
  );
}

/** What the picked run is, in one line, and what it did and did not measure, one click away. */
export function RunDetails({ runs, run }: { runs: RunInfo[]; run: string }) {
  const info = runs.find((r) => r.run === run);
  if (!info) return null;
  const k = info.self_consistency_samples ?? null;
  const conditions = info.conditions ?? null;
  const subset = info.subset_of ?? null;
  const parent = subset ? runs.find((r) => r.run === subset.run) : undefined;
  const parentLabel = parent ? `"${runLabel(parent, runs)}"` : "larger";

  const notes: { key: string; text: ReactNode }[] = [];
  if (k === 1) notes.push({ key: "k", text: "One sample per question: self-consistency could not be measured, so RQ2 leaves those signals out." });
  if (conditions && !conditions.includes("closed_book") && !conditions.includes("oracle"))
    notes.push({ key: "cond", text: "Retrieval condition only: no closed-book or oracle answers, so no comparison between conditions." });
  if (subset)
    notes.push({
      key: "subset",
      text: (
        <>
          {subset.prefix
            ? `These are the first ${fmtInt(info.n)} PubMedQA questions in dataset order of the ${parentLabel} run, not a random sample, so its numbers need not carry over to the full set.`
            : `These ${fmtInt(info.n)} questions are a subset of the ${parentLabel} run; how they were chosen is not recorded, so its numbers need not carry over to the full set.`}{" "}
          On the {fmtInt(subset.answered_in_both)} questions answered in both, the first-sample decision matches the {parentLabel} run for{" "}
          {fmtInt(subset.first_sample_matches)}.
          {k !== null && k > 1
            ? ` A question counts as failed here if any of its ${fmtInt(k)} samples was invalid${
                subset.failed_only_here > 0
                  ? `; ${fmtInt(subset.failed_only_here)} ${subset.failed_only_here === 1 ? "question failed here but was" : "questions failed here but were"} answered in the ${parentLabel} run`
                  : ""
              }.`
            : null}
          {!info.available.includes("rq3_conformal") ? " RQ3 was not run on this subset." : null}
        </>
      ),
    });
  notes.push({
    key: "descriptive",
    text: "Each run is shown on its own. Differences between generators are descriptive: no test pairs the runs, and each run's RQ2 and RQ3 are computed on splits of its own questions.",
  });

  const facts = [
    k !== null ? `${fmtInt(k)} sample${k === 1 ? "" : "s"} per question` : null,
    info.n !== null ? `${fmtInt(info.n)} questions` : null,
    failedText(info),
  ].filter(Boolean);

  return (
    <div className="mt-3 flex flex-col gap-y-1 text-[0.8125rem] text-muted sm:flex-row sm:items-start sm:gap-x-4">
      <p className="num flex min-h-9 shrink-0 flex-wrap items-center gap-x-2 gap-y-1">
        <span className="rounded-chip bg-raised px-1.5 py-px font-mono text-xs text-ink">{info.generator}</span>
        <span>{facts.join(", ")}</span>
      </p>
      <Disclosure className="min-w-0 flex-1">
        <Disclosure.Summary meta={`${notes.length} note${notes.length === 1 ? "" : "s"}`}>About this run</Disclosure.Summary>
        <Disclosure.Body>
          <ul className="max-w-[78ch] list-disc space-y-1 pl-4">
            {notes.map((note) => (
              <li key={note.key}>{note.text}</li>
            ))}
          </ul>
        </Disclosure.Body>
      </Disclosure>
    </div>
  );
}
