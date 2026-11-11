"use client";

import type { ReactNode } from "react";

import { useApi } from "@/components/shell/ApiProvider";
import { Code } from "@/components/ui/code";
import { readinessGaps, type Gap } from "@/lib/readiness";

/** What the running server is missing, said before a question fails for that reason. */
export function Readiness({ retriever, isDefault }: { retriever: string | null; isDefault: boolean }) {
  const { health, meta } = useApi();
  if (!health) return null;
  const model = meta?.models?.llm ?? null;
  const copy: Record<Gap, ReactNode> = {
    api_key: (
      <>
        No API key is configured, so answers cannot be generated. Add <Code>ANTHROPIC_API_KEY</Code> to{" "}
        <Code>.env</Code> and restart the server.
      </>
    ),
    index: (
      <>
        No search index has been built. Run <Code>primum data build kb</Code> and then{" "}
        <Code>primum index build kb</Code>.
      </>
    ),
    encoder: (
      <>
        Questions will fail at retrieval: {isDefault ? "the server's default retriever, " : "the retriever "}
        <Code>{retriever}</Code>, needs the fine-tuned encoder, which has not been trained. Run{" "}
        <Code>primum train encoder</Code>, or choose a retriever that does not, such as <Code>bm25</Code> or{" "}
        <Code>hybrid-base</Code>.
      </>
    ),
    bundle: (
      <>
        No deployment bundle exists for {model ? <Code>{model}</Code> : "the configured generator"}, so answers are
        shown without a certified threshold. A bundle comes from an RQ3 run on that generator&apos;s own records, saved
        as <Code wrap>artifacts/deployment/{model ?? "<model>"}/bundle.json</Code>.
      </>
    ),
  };
  const gaps = readinessGaps(health, retriever);
  if (gaps.length === 0) return null;
  return (
    <div className="rounded-card border border-dashed border-frame px-4 py-4 sm:px-5">
      <p className="text-sm font-medium text-ink">
        The server is running, with {gaps.length === 1 ? "one thing" : `${gaps.length} things`} not yet in place
      </p>
      <ul className="mt-1.5 space-y-1.5 text-sm leading-relaxed text-muted">
        {gaps.map((gap) => (
          <li key={gap}>{copy[gap]}</li>
        ))}
      </ul>
    </div>
  );
}
