import { Code } from "@/components/ui/code";
import { InfoGlyph } from "@/components/ui/glyphs";

interface FixtureBarProps {
  scenario: string;
  scenarios: readonly { id: string; label: string }[];
  onScenarioChange: (id: string) => void;
}

/** Always on screen while the view replays canned events, so no fixture is mistaken for output. */
export function FixtureBar({ scenario, scenarios, onScenarioChange }: FixtureBarProps) {
  return (
    <aside
      aria-label="Fixture data"
      className="mb-6 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-card bg-unsupported-bg px-3.5 py-2 text-[0.8125rem] leading-snug text-unsupported-ink ring-1 ring-inset ring-unsupported/40"
    >
      <InfoGlyph size={15} className="max-sm:hidden" />
      <p className="min-w-0 flex-1 basis-72">
        <span className="font-semibold">Fixture data. Not real output.</span> Every passage, claim and number on this
        view is invented for interface development. Unset <Code>NEXT_PUBLIC_FIXTURES</Code> to talk to the API.
      </p>
      <label className="flex items-center gap-2 font-medium">
        Scenario
        <select
          value={scenario}
          onChange={(event) => onScenarioChange(event.target.value)}
          className="h-11 rounded-control border border-unsupported/60 bg-surface px-2 text-[0.8125rem] text-ink sm:h-8"
        >
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </label>
    </aside>
  );
}
