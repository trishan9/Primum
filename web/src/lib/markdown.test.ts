import assert from "node:assert/strict";
import { test } from "node:test";

import { citedIndices, groupCitations, parseInline, parseMarkdown } from "./markdown.ts";

test("citation markers become cite nodes", () => {
  assert.deepEqual(parseInline("Aspirin lowers risk [1][3]."), [
    { kind: "text", text: "Aspirin lowers risk " },
    { kind: "cite", n: 1 },
    { kind: "cite", n: 3 },
    { kind: "text", text: "." },
  ]);
});

test("grouped and ranged citations expand", () => {
  assert.deepEqual(citedIndices("a [1, 2] b [4-6] c [2]"), [1, 2, 4, 5, 6]);
});

test("bold and italics nest citations", () => {
  assert.deepEqual(parseInline("**Strong [2]** and *soft*"), [
    { kind: "strong", children: [{ kind: "text", text: "Strong " }, { kind: "cite", n: 2 }] },
    { kind: "text", text: " and " },
    { kind: "em", children: [{ kind: "text", text: "soft" }] },
  ]);
});

test("paragraphs and lists split into blocks", () => {
  const blocks = parseMarkdown("First line\nstill first.\n\n- one [1]\n- two\n\n1. alpha\n2. beta\n");
  assert.deepEqual(
    blocks.map((b) => b.kind),
    ["p", "ul", "ol"],
  );
  assert.equal(blocks[1]?.kind === "ul" && blocks[1].items.length, 2);
});

test("angle brackets stay plain text", () => {
  assert.deepEqual(parseInline("<script>alert(1)</script>"), [{ kind: "text", text: "<script>alert(1)</script>" }]);
});

test("a bracketed year or word is not a citation", () => {
  assert.deepEqual(citedIndices("See [note] and [2024] and [12]"), [12]);
});

test("adjacent citations group with the word before them", () => {
  assert.deepEqual(groupCitations(parseInline("Aspirin lowers risk [1][3].")), [
    { kind: "text", text: "Aspirin lowers " },
    { kind: "cites", lead: "risk", ns: [1, 3] },
    { kind: "text", text: "." },
  ]);
});

test("whitespace between a citation and punctuation is dropped", () => {
  assert.deepEqual(groupCitations(parseInline("over ten years [3] . Next")), [
    { kind: "text", text: "over ten " },
    { kind: "cites", lead: "years", ns: [3] },
    { kind: "text", text: ". Next" },
  ]);
});

test("a citation at the start of a line has no lead", () => {
  assert.deepEqual(groupCitations(parseInline("[2] then text")), [
    { kind: "cites", lead: "", ns: [2] },
    { kind: "text", text: " then text" },
  ]);
});
