import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import "./globals.css";

const stackSans = Space_Grotesk({
  variable: "--font-stack-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Remember",
  description: "Hack Princeton Fall 2025",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${stackSans.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
