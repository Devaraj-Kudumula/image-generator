"use client";

import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { MobileNav } from "@/components/layout/sidebar";
import { ThemeToggle } from "@/components/theme-toggle";
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

      <ThemeToggle />
    </header>
  );
}

export function getPageTitle(pathname: string) {
  return titles[pathname] ?? "Figure Studio";
}
