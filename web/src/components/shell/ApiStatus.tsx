"use client";

import { API_URL } from "@/lib/api";

import { useApi } from "./ApiProvider";

const LABEL = {
  checking: "Checking API",
  up: "API connected",
  down: "API unreachable",
} as const;

/** The dot marks a real state: whether the last health check of the API succeeded. */
export function ApiStatus() {
  const { reachability, refresh } = useApi();
  const host = API_URL.replace(/^https?:\/\//, "");

  return (
    <button
      type="button"
      onClick={refresh}
      title={`${LABEL[reachability]} at ${host}. Select to check again.`}
      className="flex min-h-11 items-center gap-2 rounded-control px-2 text-[0.8125rem] text-muted transition-colors duration-150 hover:bg-raised hover:text-ink sm:min-h-9"
    >
      <span
        aria-hidden="true"
        className={`size-2 rounded-full ${
          reachability === "up"
            ? "bg-supported"
            : reachability === "down"
              ? "border-[1.5px] border-contradicted"
              : "border-[1.5px] border-control"
        }`}
      />
      <span role="status">{LABEL[reachability]}</span>
    </button>
  );
}
