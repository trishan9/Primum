import { cn } from "./cn";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

export interface SegmentedProps<T extends string> {
  legend: string;
  name: string;
  value: T;
  options: readonly SegmentedOption<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  /** Hides the legend visually when a visible label elsewhere already names the group. */
  hideLegend?: boolean;
  /** Stretches the track to its container below 480px, where options would otherwise wrap. */
  fill?: boolean;
  className?: string;
}

/**
 * A radio group drawn as a segmented track: the selected option is a raised white pill with a
 * 3:1 edge. Arrow keys move the selection, as with any radio group. It has no state of its own,
 * so it renders in a client parent that owns `value`.
 */
export function Segmented<T extends string>({
  legend,
  name,
  value,
  options,
  onChange,
  disabled,
  hideLegend,
  fill,
  className,
}: SegmentedProps<T>) {
  return (
    <fieldset disabled={disabled} className={cn("min-w-0", className)}>
      <legend className={hideLegend ? "sr-only" : "mb-1.5 text-[0.8125rem] font-medium text-ink"}>{legend}</legend>
      <div
        className={cn(
          "inline-flex max-w-full flex-wrap gap-0.5 rounded-control bg-raised p-0.5 ring-1 ring-line ring-inset",
          fill && "flex w-full max-[30rem]:[&>label]:flex-1",
        )}
      >
        {options.map((option) => {
          const checked = option.value === value;
          return (
            <label
              key={option.value}
              className={cn(
                "num relative flex min-h-11 min-w-11 cursor-pointer items-center justify-center rounded-[6px] px-2.5 text-sm transition-[background-color,color,box-shadow] duration-150 sm:min-h-8",
                "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
                checked
                  ? "bg-surface font-medium text-ink shadow-control ring-1 ring-control/80"
                  : "text-muted hover:bg-active hover:text-ink",
                disabled && "cursor-not-allowed opacity-70",
              )}
            >
              <input
                type="radio"
                name={name}
                value={option.value}
                checked={checked}
                onChange={() => onChange(option.value)}
                className="sr-only"
              />
              {option.label}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
