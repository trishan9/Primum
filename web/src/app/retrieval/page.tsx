import type { Metadata } from "next";

import { RetrievalLab } from "@/components/retrieval/RetrievalLab";
import { PageHeader } from "@/components/ui/page-header";

export const metadata: Metadata = { title: "Retrieval lab" };

/** The header is static and renders on the server; only the playground below needs the client. */
export default function RetrievalPage() {
  return (
    <div className="flex flex-col gap-6">
      <PageHeader>
        <PageHeader.Heading>
          <PageHeader.Title>Retrieval lab</PageHeader.Title>
          <PageHeader.Description>
            One query, several retrievers, ranked side by side. A line follows each passage from one ranking to the next,
            so you can see where lexical search, dense search and their fusion agree and where they part.
          </PageHeader.Description>
        </PageHeader.Heading>
      </PageHeader>
      <RetrievalLab />
    </div>
  );
}
