"use client";

import { Fragment, useMemo } from "react";

import { groupCitations, parseMarkdown, type Block, type Inline } from "@/lib/markdown";

interface AnswerTextProps {
  text: string;
  /** How many passages exist; a marker beyond this is shown as plain text, not a dead control. */
  passageCount: number;
  streaming?: boolean;
  onCite: (passage: number | null) => void;
  onOpenPassage: (passage: number) => void;
}

/**
 * A citation marker, set as a superscript pill: small enough that it never opens the line,
 * with a hit area larger than it looks.
 */
function CitationChip({
  n,
  valid,
  onCite,
  onOpenPassage,
}: { n: number; valid: boolean } & Pick<AnswerTextProps, "onCite" | "onOpenPassage">) {
  if (!valid) return <span className="num relative -top-[0.45em] font-sans text-[0.62em] text-muted">[{n}]</span>;
  return (
    <button
      type="button"
      onClick={() => onOpenPassage(n)}
      onMouseEnter={() => onCite(n)}
      onMouseLeave={() => onCite(null)}
      onFocus={() => onCite(n)}
      onBlur={() => onCite(null)}
      aria-label={`Source ${n}: show in sources`}
      className="num relative -top-[0.5em] inline-flex h-[1.6em] min-w-[1.6em] items-center justify-center rounded-full bg-accent-soft px-[0.4em] align-baseline font-sans text-[0.62em] font-semibold leading-none text-accent-ink transition-colors duration-(--dur-fast) before:absolute before:-inset-2.5 before:content-[''] hover:bg-accent hover:text-on-accent focus-visible:bg-accent focus-visible:text-on-accent"
    >
      {n}
    </button>
  );
}

function renderInline(nodes: Inline[], props: AnswerTextProps, keyPrefix: string): React.ReactNode {
  return groupCitations(nodes).map((node, i) => {
    const key = `${keyPrefix}-${i}`;
    switch (node.kind) {
      case "text":
        return <Fragment key={key}>{node.text}</Fragment>;
      case "strong":
        return <strong key={key}>{renderInline(node.children, props, key)}</strong>;
      case "em":
        return <em key={key}>{renderInline(node.children, props, key)}</em>;
      case "cites":
        // The markers and the word they follow wrap as one unit, so a marker never starts a line.
        return (
          <span key={key} className="whitespace-nowrap">
            {node.lead}
            <span className="ml-[0.12em] inline-flex gap-[0.18em]">
              {node.ns.map((n, j) => (
                <CitationChip
                  key={`${n}-${j}`}
                  n={n}
                  valid={n >= 1 && n <= props.passageCount}
                  onCite={props.onCite}
                  onOpenPassage={props.onOpenPassage}
                />
              ))}
            </span>
          </span>
        );
    }
  });
}

function renderBlock(block: Block, props: AnswerTextProps, key: string, caret: React.ReactNode): React.ReactNode {
  switch (block.kind) {
    case "p":
      return (
        <p key={key}>
          {renderInline(block.children, props, key)}
          {caret}
        </p>
      );
    case "heading":
      return (
        <p key={key} className="font-semibold">
          {renderInline(block.children, props, key)}
          {caret}
        </p>
      );
    case "ul":
    case "ol": {
      const Tag = block.kind;
      return (
        <Tag key={key}>
          {block.items.map((item, i) => (
            <li key={`${key}-${i}`}>
              {renderInline(item, props, `${key}-${i}`)}
              {i === block.items.length - 1 ? caret : null}
            </li>
          ))}
        </Tag>
      );
    }
  }
}

/** The answer as typeset prose. Citation markers become controls that point into the evidence column. */
export function AnswerText(props: AnswerTextProps) {
  const blocks = useMemo(() => parseMarkdown(props.text), [props.text]);
  const caret = props.streaming ? (
    <span
      aria-hidden="true"
      className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[0.18em] bg-accent"
      style={{ animation: "caret-blink 1s steps(1) infinite" }}
    />
  ) : null;

  return (
    <div className="reading max-w-[68ch] text-ink">
      {blocks.length === 0 && caret ? <p>{caret}</p> : null}
      {blocks.map((block, i) => renderBlock(block, props, `b${i}`, i === blocks.length - 1 ? caret : null))}
    </div>
  );
}
