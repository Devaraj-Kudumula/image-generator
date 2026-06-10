"use client";

import { useTheme } from "next-themes";
import { Menu, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { MobileNav } from "@/components/layout/sidebar";
import { useSidebarStore } from "@/lib/store/generation-store";

const titles: Record<string, string> = {
  "/": "Studio",
  "/edit": "Edit",
  "/gallery": "Gallery",
  "/chat": "AI Chat",
  "/docs": "Docs Q&A",
  "/settings": "Settings",
};

export function TopBar() {
  const { theme, setTheme } = useTheme();
  const { setMobileOpen } = useSidebarStore();

  return (
    <header className="flex h-14 items-center justify-between border-b bg-background/80 px-4 backdrop-blur md:px-6">
      <div className="flex items-center gap-3">
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-64 p-0">
            <div className="border-b px-4 py-4">
              <p className="font-semibold">Figure Studio</p>
            </div>
            <MobileNav onNavigate={() => setMobileOpen(false)} />
          </SheetContent>
        </Sheet>
        <div>
          <h1 className="text-sm font-semibold md:text-base">
            Scientific figures, made effortless
          </h1>
          <p className="hidden text-xs text-muted-foreground sm:block">
            Turn text, sketches, and references into publication-ready figures
          </p>
        </div>
      </div>

      <Button
        variant="ghost"
        size="icon"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        aria-label="Toggle theme"
      >
        <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
        <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
      </Button>
    </header>
  );
}

export function getPageTitle(pathname: string) {
  return titles[pathname] ?? "Figure Studio";
}
