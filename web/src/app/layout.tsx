import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, STIX_Two_Text } from "next/font/google";

import { ApiProvider } from "@/components/shell/ApiProvider";
import { AppShell } from "@/components/shell/AppShell";
import { AppSidebar } from "@/components/shell/Sidebar";
import { TopBar } from "@/components/shell/TopBar";

import "./globals.css";

// Interface, labels and every figure: a sans drawn for technical documentation, with true
// tabular figures and Greek (α, δ and λ̂ are content here).
const plexSans = IBM_Plex_Sans({
  subsets: ["latin", "latin-ext", "greek"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-sans",
  display: "swap",
});

// Answer and method prose only: STIX Two, built for scientific publishing, so an answer reads
// like a paragraph of the literature it cites.
const stix = STIX_Two_Text({
  subsets: ["latin", "latin-ext", "greek"],
  style: ["normal", "italic"],
  variable: "--font-stix",
  display: "swap",
});

// Commands, identifiers and model names only.
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
    { media: "(prefers-color-scheme: light)", color: "#eef1ef" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0c0c" },
  ],
};

// Applies the saved theme and sidebar state before first paint, so neither flashes nor shifts.
const prePaint = `try{var d=document.documentElement,t=localStorage.getItem("primum-theme");if(t==="light"||t==="dark")d.dataset.theme=t;if(localStorage.getItem("primum-sidebar")==="collapsed")d.dataset.sidebar="collapsed"}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${stix.variable} ${plexSans.variable} ${plexMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: prePaint }} />
      </head>
      <body className="font-sans text-ink antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 focus:rounded-control focus:bg-primary focus:px-3 focus:py-2 focus:text-sm focus:text-on-primary"
        >
          Skip to content
        </a>
        <ApiProvider>
          <AppShell sidebar={<AppSidebar />} topBar={<TopBar />}>
            {children}
          </AppShell>
        </ApiProvider>
      </body>
    </html>
  );
}
