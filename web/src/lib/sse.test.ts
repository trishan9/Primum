import assert from "node:assert/strict";
import { test } from "node:test";

import { createSSEParser } from "./sse.ts";

test("parses event and data fields into messages", () => {
  const parser = createSSEParser();
  const out = parser.feed('event: token\ndata: {"text":"a"}\n\nevent: answer\ndata: {"text":"ab"}\n\n');
  assert.deepEqual(out, [
    { event: "token", data: '{"text":"a"}' },
    { event: "answer", data: '{"text":"ab"}' },
  ]);
});

test("handles CRLF line endings and keep-alive comments", () => {
  const parser = createSSEParser();
  const out = parser.feed(': ping - 2026\r\n\r\nevent: stage\r\ndata: {"name":"triage"}\r\n\r\n');
  assert.deepEqual(out, [{ event: "stage", data: '{"name":"triage"}' }]);
});

test("reassembles a message split across arbitrary chunk boundaries", () => {
  const text = 'event: retrieval\r\ndata: {"retriever":"hybrid-ft","hits":[]}\r\n\r\nevent: done\r\ndata: {}\r\n\r\n';
  for (let size = 1; size <= 7; size++) {
    const parser = createSSEParser();
    const out = [];
    for (let i = 0; i < text.length; i += size) out.push(...parser.feed(text.slice(i, i + size)));
    out.push(...parser.flush());
    assert.deepEqual(
      out.map((m) => m.event),
      ["retrieval", "done"],
      `chunk size ${size}`,
    );
    assert.equal(out[0]?.data, '{"retriever":"hybrid-ft","hits":[]}');
  }
});

test("joins multi-line data with newlines and defaults the event name", () => {
  const parser = createSSEParser();
  assert.deepEqual(parser.feed("data: one\ndata: two\n\n"), [{ event: "message", data: "one\ntwo" }]);
});

test("flush emits a final message that lacks its trailing blank line", () => {
  const parser = createSSEParser();
  assert.deepEqual(parser.feed('event: error\ndata: {"kind":"budget"}'), []);
  assert.deepEqual(parser.flush(), [{ event: "error", data: '{"kind":"budget"}' }]);
});
