"use client";

import { Fragment, useMemo } from "react";

import { parseMarkdown, type Block, type Inline } from "@/lib/markdown";

interface AnswerTextProps {
  text: string;
  /** How many passages exist; a marker beyond this is shown as plain text, not a dead control. */
  passageCount: number;
  streaming?: boolean;
  onCite: (passage: number | null) => void;
  onOpenPassage: (passage: number) => void;
}

function CitationChip({ n, valid, onCite, onOpenPassage }: { n: number; valid: boolean } & Pick<AnswerTextProps, "onCite" | "onOpenPassage">) {
  if (!valid) return <span className="num font-sans text-[0.75em] text-muted">[{n}]</span>;
  return (
    <button
      type="button"
      onClick={() => onOpenPassage(n)}
      onMouseEnter={() => onCite(n)}
      onMouseLeave={() => onCite(null)}
      onFocus={() => onCite(n)}
      onBlur={() => onCite(null)}
      aria-label={`Passage ${n}: show in evidence`}
      className="num relative -top-[0.1em] mx-[0.12em] inline-flex h-[1.5em] min-w-[1.5em] items-center justify-center rounded-chip border border-control bg-surface px-[0.3em] align-baseline font-sans text-[0.68em] font-semibold leading-none text-ink transition-colors duration-150 before:absolute before:-inset-3 before:content-[''] hover:border-ink hover:bg-primary hover:text-on-primary"
    >
      {n}
    </button>
  );
}

function renderInline(nodes: Inline[], props: AnswerTextProps, keyPrefix: string): React.ReactNode {
  return nodes.map((node, i) => {
    const key = `${keyPrefix}-${i}`;
    switch (node.kind) {
      case "text":
        return <Fragment key={key}>{node.text}</Fragment>;
      case "strong":
        return <strong key={key}>{renderInline(node.children, props, key)}</strong>;
      case "em":
        return <em key={key}>{renderInline(node.children, props, key)}</em>;
      case "cite":
        return (
          <CitationChip
            key={key}
            n={node.n}
            valid={node.n >= 1 && node.n <= props.passageCount}
            onCite={props.onCite}
            onOpenPassage={props.onOpenPassage}
          />
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
      className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[0.18em] bg-ink"
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
