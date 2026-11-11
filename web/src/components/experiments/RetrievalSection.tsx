"use client";

import { useMemo } from "react";

import { deployedRetriever } from "@/lib/charts/rq1";
import { findMetricTables } from "@/lib/results";

import { LossCard, PairedDiffCard, PerQueryCard, RecallAtKCard } from "./charts/rq1-cards";
import { RawFile, Unrecognised } from "./parts";
import { groupTables } from "./readers";
import { TableGroupCard } from "./TableGroupCard";
import type { Loaded } from "./useExperiment";

const FILE = "rq1_retrieval";

/**
 * RQ1, retrieval. The absolute numbers per dataset as a forest table (every metric, with its
 * interval), then the figures that answer the question: does the gold abstract make the top k,
 * which retrievers beat the baseline query by query, what the means hide, and whether the
 * encoder's fine-tuning did anything.
 */
export function RetrievalSection({ data, training }: { data: unknown; training: Loaded<unknown> }) {
  const groups = useMemo(() => groupTables(findMetricTables(data)), [data]);
  const quality = groups.find((g) => g.key === "");
  const paired = groups.find((g) => g.key === "paired_tests");
  const others = groups.filter((g) => g !== quality && g !== paired);
  const deployed = deployedRetriever(data);
  const source = `results/${FILE}.json`;

  return (
    <div className="space-y-4">
      {groups.length === 0 ? <Unrecognised file={FILE} /> : null}
      {quality ? (
        <TableGroupCard
          tables={quality.tables}
          title="Retrieval quality by retriever"
          fallback="Retrieval"
          rowLabel="Retriever"
          variantLegend="Dataset"
          reference="bm25"
          source={source}
        />
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <RecallAtKCard data={data} deployed={deployed} />
        <PairedDiffCard data={data} deployed={deployed} />
      </div>
      <PerQueryCard data={data} deployed={deployed} />

      {others.map((group) => (
        <TableGroupCard key={group.key} tables={group.tables} fallback="Retrieval" rowLabel="Retriever" variantLegend="Table" source={source} />
      ))}

      {training.status === "ready" ? <LossCard training={training.data} /> : null}
      <RawFile data={data} name={FILE} />
    </div>
  );
}
