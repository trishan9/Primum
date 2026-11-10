import Link from "next/link";

export default function NotFound() {
  return (
    <div className="max-w-[60rem]">
      <h1 className="font-serif text-[1.75rem] font-semibold leading-tight tracking-[-0.015em] text-ink">No such page</h1>
      <p className="mt-1 max-w-[62ch] text-[0.9375rem] leading-relaxed text-muted">
        Primum has four views, and this address is not one of them.
      </p>
      <ul className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-[0.9375rem]">
        <li>
          <Link href="/" className="text-link underline underline-offset-4">
            Ask
          </Link>
        </li>
        <li>
          <Link href="/retrieval" className="text-link underline underline-offset-4">
            Retrieval lab
          </Link>
        </li>
        <li>
          <Link href="/experiments" className="text-link underline underline-offset-4">
            Experiments
          </Link>
        </li>
        <li>
          <Link href="/method" className="text-link underline underline-offset-4">
            Method
          </Link>
        </li>
      </ul>
    </div>
  );
}
