import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

type CardElement = "section" | "div" | "article" | "aside" | "li" | "ul" | "figure";
type CardVariant = "default" | "inset" | "dashed" | "accent";

const VARIANT: Record<CardVariant, string> = {
  // A white card on the sheet: one hairline, no shadow.
  default: "border border-line bg-surface",
  // A well: one tone step down, no edge. For grouped readings inside a card or on the sheet.
  inset: "bg-raised",
  // A block with nothing to show yet: it keeps its frame.
  dashed: "border border-dashed border-frame",
  // The one card on a screen that holds the answer to its question.
  accent: "border border-accent/40 bg-surface",
};

export interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: CardElement;
  variant?: CardVariant;
  children: ReactNode;
}

/**
 * A block of the canvas. Compose it:
 *
 *   <Card>
 *     <Card.Header>
 *       <Card.Heading><Card.Title>Evidence</Card.Title><Card.Description>5 passages</Card.Description></Card.Heading>
 *       <Card.Actions>…</Card.Actions>
 *     </Card.Header>
 *     <Card.Body>…</Card.Body>
 *     <Card.Footer>…</Card.Footer>
 *   </Card>
 *
 * Without sub-parts it is a plain framed box (`<Card className="p-5">`).
 */
function CardRoot({ as: Tag = "section", variant = "default", className, children, ...rest }: CardProps) {
  return (
    <Tag className={cn("min-w-0 rounded-card", VARIANT[variant], className)} {...rest}>
      {children}
    </Tag>
  );
}

function CardHeader({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 pt-4 sm:px-5", className)} {...rest}>
      {children}
    </div>
  );
}

/** Groups a title and its description so they stay together on the left of the header. */
function CardHeading({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("min-w-0 flex-1", className)}>{children}</div>;
}

function CardTitle({
  as: Tag = "h2",
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLHeadingElement> & { as?: "h2" | "h3" | "h4" | "p" }) {
  return (
    <Tag className={cn("text-[0.9375rem] font-semibold leading-6 text-ink", className)} {...rest}>
      {children}
    </Tag>
  );
}

function CardDescription({ className, children, ...rest }: HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn("mt-0.5 text-[0.8125rem] leading-[1.125rem] text-muted", className)} {...rest}>
      {children}
    </p>
  );
}

/** Controls or a period stamp, on the right of the header. */
function CardActions({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex shrink-0 flex-wrap items-center gap-2", className)}>{children}</div>;
}

function CardBody({
  className,
  flush = false,
  children,
  ...rest
}: HTMLAttributes<HTMLDivElement> & {
  /** No side padding: for a table or a list whose rows run to the card's edges. */
  flush?: boolean;
}) {
  return (
    <div className={cn(flush ? "py-3" : "px-4 py-4 sm:px-5", className)} {...rest}>
      {children}
    </div>
  );
}

function CardFooter({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-rule px-4 py-3 text-[0.8125rem] text-muted sm:px-5", className)}
      {...rest}
    >
      {children}
    </div>
  );
}

export const Card = Object.assign(CardRoot, {
  Header: CardHeader,
  Heading: CardHeading,
  Title: CardTitle,
  Description: CardDescription,
  Actions: CardActions,
  Body: CardBody,
  Footer: CardFooter,
});

export { CardActions, CardBody, CardDescription, CardFooter, CardHeader, CardHeading, CardTitle };
