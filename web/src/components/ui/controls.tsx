import type { ButtonHTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "quiet";

const VARIANT: Record<Variant, string> = {
  primary:
    "bg-primary text-on-primary hover:opacity-90 active:opacity-80 disabled:bg-raised disabled:text-muted disabled:opacity-100",
  secondary:
    "border border-control bg-surface text-ink hover:border-ink active:bg-raised disabled:border-line disabled:text-muted",
  quiet: "text-ink underline decoration-control underline-offset-4 hover:decoration-ink disabled:text-muted",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

/** 44px tall on touch screens, 36px where a pointer is precise. */
export function Button({ variant = "secondary", className, children, type = "button", ...rest }: ButtonProps) {
  const size = variant === "quiet" ? "min-h-11 px-0.5 sm:min-h-0" : "min-h-11 px-4 sm:min-h-9 sm:px-3.5";
  return (
    <button
      type={type}
      className={`inline-flex items-center justify-center gap-2 rounded-control text-sm font-medium transition-[opacity,border-color,background-color] duration-150 ${size} ${VARIANT[variant]} ${className ?? ""}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** A shell command or identifier, shown in the one place monospace belongs. */
/** Inline code. `wrap` lets a long value (a file path) break anywhere instead of overflowing a narrow screen. */
export function Code({ children, wrap = false }: { children: ReactNode; wrap?: boolean }) {
  return (
    <code
      className={`${wrap ? "[overflow-wrap:anywhere]" : "whitespace-nowrap"} rounded-chip bg-raised px-1.5 py-0.5 font-mono text-[0.8125rem] text-ink`}
    >
      {children}
    </code>
  );
}

interface SegmentedProps<T extends string> {
  legend: string;
  name: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
  /** Hides the legend visually when a visible label elsewhere already names the group. */
  hideLegend?: boolean;
}

/** A radio group drawn as a segmented control. Arrow keys move the selection, as with any radio group. */
export function Segmented<T extends string>({
  legend,
  name,
  value,
  options,
  onChange,
  disabled,
  hideLegend,
}: SegmentedProps<T>) {
  return (
    <fieldset disabled={disabled} className="min-w-0">
      <legend className={hideLegend ? "sr-only" : "mb-1.5 text-[0.8125rem] font-medium text-ink"}>{legend}</legend>
      <div className="inline-flex max-w-full flex-wrap rounded-control border border-control bg-surface p-0.5">
        {options.map((option) => {
          const checked = option.value === value;
          return (
            <label
              key={option.value}
              className={`num relative flex min-h-10 min-w-12 cursor-pointer items-center justify-center rounded-[4px] px-2.5 text-sm transition-colors duration-150 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent sm:min-h-8 ${
                checked ? "bg-primary font-medium text-on-primary" : "text-muted hover:text-ink"
              } ${disabled ? "cursor-not-allowed opacity-70" : ""}`}
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

/** A framed block of the page: a hairline edge, no shadow. */
export function Panel({
  children,
  className,
  as: Tag = "section",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article" | "aside";
} & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag className={`rounded-card border border-line bg-surface ${className ?? ""}`} {...rest}>
      {children}
    </Tag>
  );
}

/**
 * A block that has nothing to show yet. It keeps its frame and says, in one or two sentences,
 * why it is empty and what fills it. With `aside`, it spans the canvas: the reason on the left,
 * what will appear here on the right, so no half of the frame is left blank.
 */
export function EmptyBlock({
  title,
  children,
  action,
  aside,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  aside?: ReactNode;
}) {
  const body = (
    <div className="min-w-0">
      <p className="text-sm font-medium text-ink">{title}</p>
      {children ? <div className="mt-1 max-w-[62ch] text-sm leading-relaxed text-muted">{children}</div> : null}
      {action ? <div className="mt-3">{action}</div> : null}
    </div>
  );
  if (!aside) {
    return <div className="max-w-[60rem] rounded-card border border-dashed border-(--frame) px-4 py-5 sm:px-5">{body}</div>;
  }
  return (
    <div className="grid gap-x-10 gap-y-4 rounded-card border border-dashed border-(--frame) px-4 py-5 sm:px-5 lg:grid-cols-2">
      {body}
      <div className="min-w-0 max-w-[62ch] text-sm leading-relaxed text-muted">{aside}</div>
    </div>
  );
}
