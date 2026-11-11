import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "quiet";
export type ButtonSize = "sm" | "md";

const VARIANT: Record<ButtonVariant, string> = {
  // Near-black: one per screen, on the job the screen exists for.
  primary:
    "bg-primary text-on-primary shadow-control hover:opacity-90 active:opacity-80 disabled:bg-raised disabled:text-faint disabled:shadow-none disabled:opacity-100",
  secondary:
    "border border-control/70 bg-surface text-ink shadow-control hover:border-control hover:bg-raised active:bg-active disabled:border-line disabled:text-faint disabled:shadow-none",
  ghost: "text-muted hover:bg-raised hover:text-ink active:bg-active disabled:text-faint disabled:hover:bg-transparent",
  // A text action, underlined: reads as a link but does something on this page.
  quiet: "text-ink underline decoration-control underline-offset-4 hover:decoration-ink disabled:text-faint",
};

// 44px on touch screens; 32 or 36px where a pointer is precise.
const SIZE: Record<ButtonSize, string> = {
  sm: "min-h-11 px-3 text-[0.8125rem] sm:min-h-8 sm:px-2.5",
  md: "min-h-11 px-4 text-sm sm:min-h-9 sm:px-3.5",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  children: ReactNode;
}

/** Props for styling a link the same way as a button: `<Link className={buttonClass("primary")}>`. */
export function buttonClass(variant: ButtonVariant = "secondary", size: ButtonSize = "md", className?: string): string {
  const sizing = variant === "quiet" ? "min-h-11 px-0.5 text-sm sm:min-h-0" : SIZE[size];
  return cn(
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-control font-medium transition-[opacity,border-color,background-color,color] duration-150",
    sizing,
    VARIANT[variant],
    className,
  );
}

export function Button({ variant = "secondary", size = "md", className, children, type = "button", ...rest }: ButtonProps) {
  return (
    <button type={type} className={buttonClass(variant, size, className)} {...rest}>
      {children}
    </button>
  );
}

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "aria-label"> {
  /** The accessible name, also shown as the native tooltip. Required: an icon alone names nothing. */
  label: string;
  children: ReactNode;
}

/** A square button holding one glyph. 44px on touch screens, 32px with a precise pointer. */
export function IconButton({ label, className, children, type = "button", title, ...rest }: IconButtonProps) {
  return (
    <button
      type={type}
      aria-label={label}
      title={title ?? label}
      className={cn(
        "inline-flex size-11 shrink-0 items-center justify-center rounded-control text-muted transition-colors duration-150 hover:bg-active hover:text-ink disabled:text-faint lg:size-8",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}
