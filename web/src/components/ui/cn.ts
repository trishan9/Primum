/** Joins class names, dropping empty ones. Kept tiny on purpose: no merge logic, so order is the author's. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}
