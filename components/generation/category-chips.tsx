"use client";

import { cn } from "@/lib/utils";

const CATEGORIES = [
  "Biology",
  "Medicine",
  "Chemistry",
  "Protocols",
  "STEM Education",
  "CS",
  "Physics",
];

interface CategoryChipsProps {
  selected: string | null;
  onSelect: (category: string | null) => void;
}

export function CategoryChips({ selected, onSelect }: CategoryChipsProps) {
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {CATEGORIES.map((category) => (
        <button
          key={category}
          type="button"
          onClick={() => onSelect(selected === category ? null : category)}
          className={cn(
            "rounded-full border px-4 py-1.5 text-sm transition-colors",
            selected === category
              ? "border-primary bg-primary text-primary-foreground"
              : "border-border bg-background text-muted-foreground hover:border-primary/40 hover:text-foreground"
          )}
        >
          {category}
        </button>
      ))}
    </div>
  );
}

export function categoryPromptPrefix(category: string | null) {
  if (!category) return "";
  return `Create a publication-ready ${category.toLowerCase()} scientific figure: `;
}
