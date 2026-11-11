import type { ReactNode } from "react";

import { Code } from "@/components/ui/code";
import { EmptyBlock } from "@/components/ui/empty-state";
import { fmtInt } from "@/lib/format";
import { readProvenance } from "@/lib/results";

import { Disclosure } from "@/components/ui/disclosure";

/** "written 2026-11-10 16:10 UTC" from an ISO stamp; anything else is shown as written. */
export function fmtWritten(created: string | null): string | null {
  if (!created) return null;
  const m = created.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
  return m ? `${m[1]} ${m[2]}${/(\+00:00|Z)$/.test(created) ? " UTC" : ""}` : created;
}

/** Where a results file came from: its path, when it was written, n, and the models named in its meta. */
export function Provenance({ data, file, children }: { data: unknown; file: string; children?: ReactNode }) {
  const p = readProvenance(data);
  return (
    <dl className="grid gap-x-6 gap-y-1.5 sm:grid-cols-[max-content_minmax(0,1fr)]">
      <dt className="text-faint">File</dt>
      <dd className="min-w-0 font-mono text-xs text-ink [overflow-wrap:anywhere]">results/{file}.json</dd>
      {p.created ? (
        <>
          <dt className="text-faint">Written</dt>
          <dd className="num text-ink">{fmtWritten(p.created)}</dd>
        </>
      ) : null}
      {p.n !== null ? (
        <>
          <dt className="text-faint">Questions</dt>
          <dd className="num text-ink">n = {fmtInt(p.n)}</dd>
        </>
      ) : null}
      {p.models.length > 0 ? (
        <>
          <dt className="text-faint">Models</dt>
          <dd className="min-w-0">
            <ul className="space-y-0.5">
              {p.models.map((model) => (
                <li key={model} className="font-mono text-xs text-ink [overflow-wrap:anywhere]">
                  {model}
                </li>
              ))}
            </ul>
          </dd>
        </>
      ) : null}
      {children}
    </dl>
  );
}

/** The results file itself, unaltered, so a layout this page cannot read is still inspectable. */
export function RawFile({ data, name }: { data: unknown; name: string }) {
  const text = JSON.stringify(data, null, 2) ?? "";
  const lines = text.split("\n");
  const limit = 400;
  return (
    <Disclosure variant="framed">
      <Disclosure.Summary meta={`${fmtInt(lines.length)} lines`}>Raw results file, results/{name}.json</Disclosure.Summary>
      <Disclosure.Body className="p-0! pl-0!">
        <pre tabIndex={0} role="region" aria-label={`Raw results file, ${name}.json`} className="relative max-h-96 overflow-auto px-4 py-3 font-mono text-xs leading-relaxed text-ink sm:px-5">
          {lines.slice(0, limit).join("\n")}
          {lines.length > limit ? `\n… ${fmtInt(lines.length - limit)} more lines in the file` : ""}
        </pre>
      </Disclosure.Body>
    </Disclosure>
  );
}

export function Unrecognised({ file }: { file: string }) {
  return (
    <EmptyBlock title="The results file is here, but this page does not recognise its layout">
      <p>
        No table, curve or per-level record could be read from <Code wrap>results/{file}.json</Code>, so no figure is drawn. The file itself is
        shown below, unaltered.
      </p>
    </EmptyBlock>
  );
}

