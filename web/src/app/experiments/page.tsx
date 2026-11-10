import type { Metadata } from "next";

import { ExperimentsView } from "@/components/experiments/ExperimentsView";

export const metadata: Metadata = { title: "Experiments" };

export default function ExperimentsPage() {
  return <ExperimentsView />;
}
