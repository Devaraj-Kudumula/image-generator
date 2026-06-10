"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Dna } from "lucide-react";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

const MODELS = [
  { id: "gemini-3-pro-image-preview", label: "Gemini 3 Pro Image" },
  { id: "gemini-3.1-flash-image-preview", label: "Gemini 3.1 Flash Image" },
];

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-8 w-[180px] border-none bg-muted/60 text-xs shadow-none">
        <div className="flex items-center gap-2">
          <Dna className="h-3.5 w-3.5 text-primary" />
          <SelectValue />
        </div>
      </SelectTrigger>
      <SelectContent>
        {MODELS.map((model) => (
          <SelectItem key={model.id} value={model.id}>
            {model.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
