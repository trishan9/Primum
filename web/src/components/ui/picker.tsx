"use client";

import { useId } from "react";

import { cn } from "@/components/ui/cn";
import { ChevronGlyph } from "@/components/ui/glyphs";
import { Segmented, type SegmentedOption } from "@/components/ui/segmented";

interface PickerProps<T extends string> {
  /** What is being picked; the accessible name of both forms. */
  legend: string;
  value: T;
  options: readonly SegmentedOption<T>[];
  onChange: (value: T) => void;
  /** Show the legend as a visible label beside the control. */
  showLabel?: boolean;
  /**
   * The narrowest screen at which the segmented track is used; below it the same choice is a
   * native select, which never wraps. "never" keeps the select at every width (long labels).
   */
  segmentedFrom?: "sm" | "md" | "lg" | "never";
  className?: string;
}

const SEGMENTED_SHOW = { sm: "hidden sm:block", md: "hidden md:block", lg: "hidden lg:block", never: "hidden" } as const;
const SELECT_HIDE = { sm: "sm:hidden", md: "md:hidden", lg: "lg:hidden", never: "" } as const;

/**
 * One choice, drawn to fit the width it has: a segmented track where its options fit on one line,
 * a native select where they would wrap. Only one of the two is ever displayed, so assistive
 * technology meets a single control.
 */
export function Picker<T extends string>({ legend, value, options, onChange, showLabel = false, segmentedFrom = "md", className }: PickerProps<T>) {
  const id = useId();
  return (
    <div className={cn("flex min-w-0 items-center gap-2.5", className)}>
      {showLabel ? (
        <span aria-hidden="true" className="shrink-0 text-[0.8125rem] text-muted">
          {legend}
        </span>
      ) : null}
      <Segmented
        legend={legend}
        hideLegend
        name={`${id}-segmented`}
        value={value}
        options={options}
        onChange={onChange}
        className={SEGMENTED_SHOW[segmentedFrom]}
      />
      <div className={cn("relative min-w-0 max-w-full", SELECT_HIDE[segmentedFrom])}>
        <label htmlFor={`${id}-select`} className="sr-only">
          {legend}
        </label>
        <select
          id={`${id}-select`}
          value={value}
          onChange={(event) => onChange(event.target.value as T)}
          className="num h-11 w-full min-w-0 max-w-full appearance-none truncate rounded-control border border-control bg-surface pl-3 pr-9 text-sm text-ink sm:h-9"
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronGlyph size={14} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted" />
      </div>
    </div>
  );
}
