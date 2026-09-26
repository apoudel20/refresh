import "./globals.css";

export const metadata = {
  title: "Refresh — Blender reconstruction",
  description: "A calm workspace for Blender reconstruction runs.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
