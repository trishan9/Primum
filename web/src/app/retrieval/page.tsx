import type { Metadata } from "next";

import { RetrievalLab } from "@/components/retrieval/RetrievalLab";

export const metadata: Metadata = { title: "Retrieval lab" };

export default function RetrievalPage() {
  return <RetrievalLab />;
}
