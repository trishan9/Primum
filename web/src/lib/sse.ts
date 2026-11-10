/**
 * Server-sent events over `fetch`. `EventSource` cannot POST, so the stream from
 * `POST /api/ask` is parsed here from the response body.
 */

export interface SSEMessage {
  event: string;
  data: string;
}

/**
 * Incremental parser for the `text/event-stream` format.
 *
 * Feed it decoded text in whatever chunks the network delivers; it returns the
 * messages completed by that chunk and keeps any partial message for the next call.
 * Lines may end in LF, CRLF or CR, and a chunk boundary may fall inside a CRLF pair.
 * Comment lines (`: ping`, which sse-starlette sends as a keep-alive) are dropped.
 */
export function createSSEParser(): { feed(chunk: string): SSEMessage[]; flush(): SSEMessage[] } {
  let buffer = "";
  let event = "";
  let data: string[] = [];

  function takeLine(line: string, out: SSEMessage[]): void {
    if (line === "") {
      if (data.length > 0) out.push({ event: event || "message", data: data.join("\n") });
      event = "";
      data = [];
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }

  function drain(final: boolean): SSEMessage[] {
    const out: SSEMessage[] = [];
    let start = 0;
    for (let i = 0; i < buffer.length; i++) {
      const ch = buffer[i];
      if (ch !== "\n" && ch !== "\r") continue;
      if (ch === "\r") {
        // A lone CR at the very end may be the first half of a CRLF split across chunks.
        if (i === buffer.length - 1 && !final) break;
        takeLine(buffer.slice(start, i), out);
        if (buffer[i + 1] === "\n") i++;
      } else {
        takeLine(buffer.slice(start, i), out);
      }
      start = i + 1;
    }
    buffer = buffer.slice(start);
    if (final) {
      if (buffer !== "") takeLine(buffer, out);
      buffer = "";
      takeLine("", out);
    }
    return out;
  }

  return {
    feed(chunk: string) {
      buffer += chunk;
      return drain(false);
    },
    flush() {
      return drain(true);
    },
  };
}

/** Reads a streaming response body and yields each SSE message as it completes. */
export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = createSSEParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const message of parser.feed(decoder.decode(value, { stream: true }))) yield message;
    }
    for (const message of parser.feed(decoder.decode())) yield message;
    for (const message of parser.flush()) yield message;
  } finally {
    reader.releaseLock();
  }
}
