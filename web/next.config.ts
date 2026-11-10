import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // `next dev` otherwise writes an AGENTS.md into the project on its own.
  agentRules: false,
};

export default nextConfig;
