/**
 * A deliberately small Markdown reader for generated answers: paragraphs, lists,
 * bold, italics, and `[n]` citation markers. Everything is parsed into data and rendered
 * by React, so model output never reaches the DOM as HTML.
 */

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "strong"; children: Inline[] }
  | { kind: "em"; children: Inline[] }
  | { kind: "cite"; n: number };

export type Block =
  | { kind: "p"; children: Inline[] }
  | { kind: "heading"; children: Inline[] }
  | { kind: "ul"; items: Inline[][] }
  | { kind: "ol"; items: Inline[][] };

/** `[1]`, `[1, 3]`, `[2-4]` and `[1][2]` all cite passages by 1-based index. */
const CITE = /\[(\d{1,3}(?:\s*[,;-]\s*\d{1,3})*)\]/g;

function expandCitation(body: string): number[] {
  const out: number[] = [];
  for (const part of body.split(/[,;]/)) {
    const range = part.trim().match(/^(\d+)\s*-\s*(\d+)$/);
    if (range) {
      const from = Number(range[1]);
      const to = Number(range[2]);
      if (to >= from && to - from <= 20) for (let n = from; n <= to; n++) out.push(n);
      continue;
    }
    const n = Number(part.trim());
    if (Number.isInteger(n) && n > 0) out.push(n);
  }
  return out;
}

function splitCitations(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(CITE)) {
    const index = match.index ?? 0;
    if (index > last) out.push({ kind: "text", text: text.slice(last, index) });
    for (const n of expandCitation(match[1] ?? "")) out.push({ kind: "cite", n });
    last = index + match[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

const EMPHASIS = /(\*\*|__)(?=\S)([\s\S]+?)(?<=\S)\1|(\*|_)(?=\S)([^*_]+?)(?<=\S)\3/g;

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(EMPHASIS)) {
    const index = match.index ?? 0;
    if (index > last) out.push(...splitCitations(text.slice(last, index)));
    if (match[1]) out.push({ kind: "strong", children: parseInline(match[2] ?? "") });
    else out.push({ kind: "em", children: splitCitations(match[4] ?? "") });
    last = index + match[0].length;
  }
  if (last < text.length) out.push(...splitCitations(text.slice(last)));
  return out;
}

export function parseMarkdown(source: string): Block[] {
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  let list: { kind: "ul" | "ol"; items: string[] } | null = null;

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ kind: "p", children: parseInline(paragraph.join(" ")) });
    paragraph = [];
  };
  const flushList = () => {
    if (list) blocks.push({ kind: list.kind, items: list.items.map(parseInline) });
    list = null;
  };

  for (const raw of source.replace(/\r\n?/g, "\n").split("\n")) {
    const line = raw.trimEnd();
    if (line.trim() === "") {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ kind: "heading", children: parseInline(heading[1] ?? "") });
      continue;
    }
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const ordered = line.match(/^\s*\d{1,2}[.)]\s+(.*)$/);
    if (bullet || ordered) {
      flushParagraph();
      const kind = bullet ? "ul" : "ol";
      if (list && list.kind !== kind) flushList();
      if (!list) list = { kind, items: [] };
      list.items.push((bullet ?? ordered)?.[1] ?? "");
      continue;
    }
    if (list && /^\s{2,}\S/.test(raw) && list.items.length) {
      list.items[list.items.length - 1] += ` ${line.trim()}`;
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  flushParagraph();
  flushList();
  return blocks;
}

/** Every passage index cited anywhere in the text, in order of first appearance. */
export function citedIndices(source: string): number[] {
  const seen = new Set<number>();
  for (const match of source.matchAll(CITE)) for (const n of expandCitation(match[1] ?? "")) seen.add(n);
  return [...seen];
}
