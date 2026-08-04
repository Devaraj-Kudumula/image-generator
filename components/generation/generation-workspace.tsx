"use client";

import { ModeTabs } from "@/components/generation/mode-tabs";
import { PromptInput } from "@/components/generation/prompt-input";
import { CategoryChips } from "@/components/generation/category-chips";
import { ResultGrid } from "@/components/generation/result-grid";
import { Alert } from "@/components/ui/alert";
import { useGenerationStore } from "@/lib/store/generation-store";

export function GenerationWorkspace() {
  const {
    mode,
    setMode,
    category,
    setCategory,
    results,
    isGenerating,
    error,
  } = useGenerationStore();

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-8 md:px-6 md:py-12">
      <div className="space-y-6 text-center">
        <div className="space-y-2">
          <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">
            Scientific figures, made effortless
          </h2>
          <p className="text-sm text-muted-foreground md:text-base">
            Turn text, sketches, and reference images into editable, publication-ready figures
          </p>
        </div>
        <ModeTabs mode={mode} onChange={setMode} />
      </div>

      <PromptInput />

      <CategoryChips selected={category} onSelect={setCategory} />

      {error && (
        <Alert className="mx-auto max-w-4xl border-destructive/50 text-destructive">
          {error}
        </Alert>
      )}

      <ResultGrid images={results} isGenerating={isGenerating} />
    </div>
  );
}
