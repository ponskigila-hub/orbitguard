import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OGB — OrbitalGuard",
  description:
    "OrbitalGuard (OGB) — AI Visual & Orbital Mission Guardian. " +
    "Decision-support system for space operators. IBM Bob AI Builders Challenge 2025.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#060d18] text-[#e2e8f0]">{children}</body>
    </html>
  );
}
