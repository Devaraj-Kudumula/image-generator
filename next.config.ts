import type { NextConfig } from "next";

const flaskBackend =
  process.env.FLASK_BACKEND_URL || "http://127.0.0.1:5002";

/** Flask API paths proxied to the Python backend in local dev. */
const FLASK_API_PATHS = [
  "generate-image",
  "edit-image",
  "get-accurate",
  "refined-prompt-image",
  "vectorize-image",
  "refine-svg-codegen",
  "ai-chat-themes",
  "ai-chat-message",
  "health",
  "doc-names",
  "session/reset",
  "chat-with-docs",
  "upload-doc",
];

const nextConfig: NextConfig = {
  async rewrites() {
    if (process.env.VERCEL) {
      // On Vercel, vercel.json rewrites Flask paths to api/index.py
      return [];
    }
    return [
      ...FLASK_API_PATHS.map((path) => ({
        source: `/api/${path}`,
        destination: `${flaskBackend}/api/${path}`,
      })),
      {
        source: "/api/images/:path*",
        destination: `${flaskBackend}/api/images/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "**" },
      { protocol: "http", hostname: "localhost" },
      { protocol: "http", hostname: "127.0.0.1" },
    ],
    unoptimized: true,
  },
};

export default nextConfig;
