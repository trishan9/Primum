import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, STIX_Two_Text } from "next/font/google";

import { ApiProvider } from "@/components/shell/ApiProvider";
import { Masthead } from "@/components/shell/Masthead";

import "./globals.css";

// Reading text is set in STIX Two, the typeface family built for scientific publishing,
// so an answer reads like a paragraph of the literature it cites.
const stix = STIX_Two_Text({
  subsets: ["latin", "latin-ext", "greek"],
  style: ["normal", "italic"],
  variable: "--font-stix",
  display: "swap",
});

// Instrument chrome and every figure: a sans drawn for technical documentation, with true tabular figures.
const plexSans = IBM_Plex_Sans({
  subsets: ["latin", "latin-ext", "greek"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-sans",
  display: "swap",
});

// Commands and identifiers only.
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "Primum", template: "%s | Primum" },
  description:
    "A research prototype for clinical question answering that shows its evidence, checks its own claims, and withholds an answer whose confidence falls below a threshold certified on a benchmark.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f3f5f4" },
    { media: "(prefers-color-scheme: dark)", color: "#0d0e0e" },
  ],
};

// Applies a saved theme before first paint so the page never flashes the wrong one.
const themeScript = `try{var t=localStorage.getItem("primum-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${stix.variable} ${plexSans.variable} ${plexMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="font-sans text-ink antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 focus:rounded-control focus:bg-primary focus:px-3 focus:py-2 focus:text-sm focus:text-on-primary"
        >
          Skip to content
        </a>
        <ApiProvider>
          <Masthead />
          <main id="main" className="mx-auto w-full max-w-[1320px] px-4 pb-24 pt-6 sm:px-6 lg:px-8 lg:pt-8">
            {children}
          </main>
        </ApiProvider>
      </body>
    </html>
  );
}
