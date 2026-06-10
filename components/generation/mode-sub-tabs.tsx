"use client";

import { cn } from "@/lib/utils";
import type { InputMode } from "@/lib/store/generation-store";

const MODES: { id: InputMode; label: string }[] = [
  { id: "generate", label: "Generate Figure" },
  { id: "enhance", label: "Enhance Figure" },
  { id: "sketch", label: "Sketch to Figure" },
  { id: "reference", label: "Add Ref Figure" },
];

interface ModeSubTabsProps {
  value: InputMode;
  onChange: (mode: InputMode) => void;
}

export function ModeSubTabs({ value, onChange }: ModeSubTabsProps) {
  return (
    <div className="flex flex-wrap gap-2 border-b pb-4">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          type="button"
          onClick={() => onChange(mode.id)}
          className={cn(
            "rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
            value === mode.id
              ? "border-primary bg-primary/5 text-foreground"
              : "border-transparent bg-muted/50 text-muted-foreground hover:text-foreground"
          )}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
}
