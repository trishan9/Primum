import Link from "next/link";

import { PageHeader } from "@/components/ui/page-header";
import { ArrowRightGlyph, BookGlyph, ChatGlyph, ColumnsGlyph, IntervalGlyph } from "@/components/ui/glyphs";

const VIEWS = [
  { href: "/", label: "Ask", glyph: ChatGlyph, text: "Ask a question and watch the pipeline decide whether to answer." },
  { href: "/retrieval", label: "Retrieval lab", glyph: ColumnsGlyph, text: "Compare lexical, dense and fused rankings for one query." },
  { href: "/experiments", label: "Experiments", glyph: IntervalGlyph, text: "The results files behind the certified threshold." },
  { href: "/method", label: "Method", glyph: BookGlyph, text: "How the system decides to answer, step by step." },
] as const;

export default function NotFound() {
  return (
    <div className="max-w-[60rem]">
      <PageHeader>
        <PageHeader.Heading>
          <PageHeader.Title>No such page</PageHeader.Title>
          <PageHeader.Description>Primum has four views, and this address is not one of them.</PageHeader.Description>
        </PageHeader.Heading>
      </PageHeader>
      <ul className="mt-6 grid gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-2">
        {VIEWS.map(({ href, label, glyph: Glyph, text }) => (
          <li key={href} className="bg-surface">
            <Link
              href={href}
              className="group flex h-full min-h-11 items-start gap-3 px-4 py-4 transition-colors duration-(--dur-fast) hover:bg-ground sm:px-5"
            >
              <Glyph className="mt-0.5 text-muted group-hover:text-ink" />
              <span className="min-w-0 flex-1">
                <span className="block text-[0.9375rem] font-semibold text-ink">{label}</span>
                <span className="mt-0.5 block text-sm leading-relaxed text-muted">{text}</span>
              </span>
              <ArrowRightGlyph className="mt-1 text-faint transition-transform duration-(--dur-fast) group-hover:translate-x-0.5 group-hover:text-ink" />
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
