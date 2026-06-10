"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LayoutTemplate } from "lucide-react";
import type { AspectRatio } from "@/lib/api";

interface FormatSelectorProps {
  aspectRatio: AspectRatio;
  style: string;
  onAspectRatioChange: (value: AspectRatio) => void;
  onStyleChange: (value: string) => void;
}

const ASPECT_RATIOS: AspectRatio[] = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4"];
const STYLES = ["flat", "realistic", "diagram", "histology"];

export function FormatSelector({
  aspectRatio,
  style,
  onAspectRatioChange,
  onStyleChange,
}: FormatSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <Select value={style} onValueChange={onStyleChange}>
        <SelectTrigger className="h-8 w-[90px] border-none bg-muted/60 text-xs shadow-none">
          <LayoutTemplate className="mr-1 h-3.5 w-3.5" />
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {STYLES.map((s) => (
            <SelectItem key={s} value={s}>
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select value={aspectRatio} onValueChange={(v) => onAspectRatioChange(v as AspectRatio)}>
        <SelectTrigger className="h-8 w-[80px] border-none bg-muted/60 text-xs shadow-none">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {ASPECT_RATIOS.map((ratio) => (
            <SelectItem key={ratio} value={ratio}>
              {ratio === "auto" ? "Auto" : ratio}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
