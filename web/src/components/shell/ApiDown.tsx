"use client";

import { Button, Code, EmptyBlock } from "@/components/ui/controls";
import { API_URL } from "@/lib/api";

import { useApi } from "./ApiProvider";

/** Said the same way on every page: where the API was expected, and how to start it. */
export function ApiDown({ what, aside }: { what: string; aside?: React.ReactNode }) {
  const { refresh, reachability } = useApi();
  return (
    <EmptyBlock
      aside={aside}
      title={`The Primum API is not answering at ${API_URL.replace(/^https?:\/\//, "")}`}
      action={
        <Button onClick={refresh} disabled={reachability === "checking"}>
          Check again
        </Button>
      }
    >
      <p>
        {what} Start the server from the project root with <Code>primum serve</Code>, or set{" "}
        <Code>NEXT_PUBLIC_API_URL</Code> if it runs somewhere else.
      </p>
    </EmptyBlock>
  );
}
