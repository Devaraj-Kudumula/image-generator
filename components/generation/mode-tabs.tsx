"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { GenerationMode } from "@/lib/store/generation-store";

interface ModeTabsProps {
  mode: GenerationMode;
  onChange: (mode: GenerationMode) => void;
}

export function ModeTabs({ mode, onChange }: ModeTabsProps) {
  return (
    <div className="flex items-center justify-center gap-2">
      <button
        type="button"
        onClick={() => onChange("illustration")}
        className={cn(
          "rounded-full px-5 py-2 text-sm font-medium transition-colors",
          mode === "illustration"
            ? "bg-primary text-primary-foreground shadow-sm"
            : "bg-muted text-muted-foreground hover:text-foreground"
        )}
      >
        Scientific Illustration
      </button>
      <button
        type="button"
        onClick={() => onChange("flowchart")}
        className={cn(
          "inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm font-medium transition-colors",
          mode === "flowchart"
            ? "bg-primary text-primary-foreground shadow-sm"
            : "bg-muted text-muted-foreground hover:text-foreground"
        )}
      >
        Flowcharts
        <Badge variant="secondary" className="rounded-full px-2 py-0 text-[10px]">
          Beta
        </Badge>
      </button>
    </div>
  );
}
