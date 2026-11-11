import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // `next dev` otherwise writes an AGENTS.md into the project on its own.
  agentRules: false,
  // Keep the dev badge off the sidebar's System card, which carries the fixture notice.
  devIndicators: { position: "bottom-right" },
};

export default nextConfig;
