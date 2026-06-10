"use client";

import { useRef, useState, useEffect } from "react";
import {
  ArrowUp,
  Paperclip,
  Lightbulb,
  Palette,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { ModeSubTabs } from "@/components/generation/mode-sub-tabs";
import { ModelSelector } from "@/components/generation/model-selector";
import { FormatSelector } from "@/components/generation/format-selector";
import { useGenerationStore } from "@/lib/store/generation-store";
import { api } from "@/lib/api";
import { useGalleryStore } from "@/lib/store/generation-store";
import { categoryPromptPrefix } from "@/components/generation/category-chips";
import { cn } from "@/lib/utils";

interface PromptInputProps {
  onGenerated?: () => void;
}

const STYLE_OPTIONS = ["flat", "realistic", "diagram", "histology"] as const;

const STYLE_PROMPT_PREFIX: Record<(typeof STYLE_OPTIONS)[number], string> = {
  flat: "",
  realistic: "Photorealistic medical illustration: ",
  diagram: "Scientific diagram style: ",
  histology: "Histology microscopy slide style: ",
};

function applyStylePrefix(prompt: string, styleValue: string): string {
  const prefix =
    STYLE_PROMPT_PREFIX[styleValue as (typeof STYLE_OPTIONS)[number]] ?? "";
  return prefix ? `${prefix}${prompt}` : prompt;
}

export function PromptInput({ onGenerated }: PromptInputProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const styleMenuRef = useRef<HTMLDivElement>(null);
  const [styleMenuOpen, setStyleMenuOpen] = useState(false);
  const {
    prompt,
    setPrompt,
    aspectRatio,
    setAspectRatio,
    style,
    setStyle,
    model,
    setModel,
    inputMode,
    setInputMode,
    category,
    isGenerating,
    setGenerating,
    setError,
    addResult,
    referenceDataUrl,
    setReferenceFile,
  } = useGenerationStore();
  const addToGallery = useGalleryStore((s) => s.addImage);

  useEffect(() => {
    if (!styleMenuOpen) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (
        styleMenuRef.current &&
        !styleMenuRef.current.contains(event.target as Node)
      ) {
        setStyleMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [styleMenuOpen]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setReferenceFile(file, reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleSubmit = async () => {
    const trimmed = prompt.trim();
    if (!trimmed && inputMode === "generate") {
      setError("Please enter a prompt");
      return;
    }

    setGenerating(true);
    setError(null);

    try {
      const categoryPrompt = `${categoryPromptPrefix(category)}${trimmed}`.trim();
      const fullPrompt = applyStylePrefix(categoryPrompt, style);
      let result;

      if (inputMode === "enhance" || inputMode === "sketch" || inputMode === "reference") {
        if (!referenceDataUrl) {
          setError("Please attach a reference or sketch image first");
          setGenerating(false);
          return;
        }
        const instructions =
          inputMode === "enhance"
            ? `Enhance this scientific figure: ${fullPrompt}`
            : inputMode === "sketch"
              ? `Convert this sketch into a publication-ready scientific figure: ${fullPrompt}`
              : `Incorporate this reference figure into a new illustration: ${fullPrompt}`;

        result = await api.editImage({
          image_data_url: referenceDataUrl,
          changes: instructions,
        });
      } else {
        result = await api.generateImage({
          prompt: fullPrompt,
          aspect_ratio: aspectRatio,
          model,
        });
      }

      const image = addResult({
        filename: result.filename,
        imageUrl: result.image_url,
        imageDataUrl: result.image_data_url,
        prompt: fullPrompt,
        aspectRatio: result.aspect_ratio ?? aspectRatio,
        kind: inputMode === "generate" ? "generated" : "edited",
      });
      addToGallery(image);
      onGenerated?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  };

  const placeholders: Record<string, string> = {
    generate: "Describe the scientific figure you want to create…",
    enhance: "Describe how to enhance the attached figure…",
    sketch: "Describe the final figure based on your sketch…",
    reference: "Describe how to use the reference figure…",
  };

  return (
    <Card className="mx-auto w-full max-w-4xl border-border/80 bg-card p-4 shadow-sm md:p-6">
      <ModeSubTabs value={inputMode} onChange={setInputMode} />

      <div className="mt-4 space-y-4">
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={placeholders[inputMode]}
          className="min-h-[120px] resize-none border-none bg-transparent px-0 text-base shadow-none focus-visible:ring-0"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void handleSubmit();
            }
          }}
        />

        {referenceDataUrl && (
          <div className="flex items-center gap-2 rounded-lg bg-muted/50 p-2 text-xs text-muted-foreground">
            <Paperclip className="h-3.5 w-3.5" />
            Reference image attached
            <button
              type="button"
              className="ml-auto text-foreground underline"
              onClick={() => setReferenceFile(null, null)}
            >
              Remove
            </button>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
          <div className="flex items-center gap-1">
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => fileRef.current?.click()}
              title="Attach image"
            >
              <Paperclip className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() =>
                setPrompt(
                  "A clean, labeled cross-section diagram showing cellular structures with accurate anatomical orientation for a medical textbook."
                )
              }
              title="Prompt suggestion"
            >
              <Lightbulb className="h-4 w-4" />
            </Button>
            <div className="relative" ref={styleMenuRef}>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className={cn("h-8 w-8", styleMenuOpen && "bg-muted")}
                title="Quick style"
                onClick={() => setStyleMenuOpen((open) => !open)}
              >
                <Palette className="h-4 w-4" />
              </Button>
              {styleMenuOpen && (
                <div className="absolute bottom-full left-0 z-50 mb-2 w-44 rounded-lg border bg-popover p-2 shadow-md">
                  <p className="mb-2 px-1 text-xs font-medium text-muted-foreground">
                    Style
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {STYLE_OPTIONS.map((option) => (
                      <button
                        key={option}
                        type="button"
                        onClick={() => {
                          setStyle(option);
                          setStyleMenuOpen(false);
                        }}
                        className={cn(
                          "rounded-md px-2 py-1 text-xs capitalize transition-colors",
                          style === option
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted/60 text-muted-foreground hover:text-foreground"
                        )}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ModelSelector value={model} onChange={setModel} />
            <FormatSelector
              aspectRatio={aspectRatio}
              style={style}
              onAspectRatioChange={setAspectRatio}
              onStyleChange={setStyle}
            />
            <Button
              type="button"
              size="icon"
              className={cn("h-9 w-9 rounded-full", isGenerating && "opacity-70")}
              disabled={isGenerating}
              onClick={() => void handleSubmit()}
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
