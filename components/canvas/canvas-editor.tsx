"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Download,
  Loader2,
  Minus,
  Plus,
  Sparkles,
  Trash2,
  Type,
} from "lucide-react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { GeneratedImage } from "@/lib/api";
import { api, resolveImageSrc } from "@/lib/api";

interface CanvasEditorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  image: GeneratedImage;
  onSaveToChat?: (pngDataUrl: string) => void;
}

type FabricModule = {
  Canvas: new (
    element: HTMLCanvasElement,
    options?: Record<string, unknown>
  ) => FabricCanvas;
  IText: new (text: string, options?: Record<string, unknown>) => FabricObject;
  Point: new (x: number, y: number) => { x: number; y: number };
  loadSVGFromString: (
    string: string,
    callback: (
      objects: FabricObject[],
      options: Record<string, unknown>
    ) => void,
    reviver?: (
      element: Element | null,
      obj: FabricObject
    ) => void
  ) => void;
};

type FabricObject = {
  type?: string;
  text?: string;
  left?: number;
  top?: number;
  fill?: string;
  stroke?: string;
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: string;
  scaleX?: number;
  scaleY?: number;
  angle?: number;
  originX?: string;
  originY?: string;
  selectable?: boolean;
  evented?: boolean;
  isEditing?: boolean;
  _ocrTargetWidth?: number;
  set: (key: string | Record<string, unknown>, value?: unknown) => void;
  getObjects?: () => FabricObject[];
  getScaledWidth?: () => number;
  enterEditing?: () => void;
  selectAll?: () => void;
};

type FabricCanvas = {
  dispose: () => void;
  add: (obj: FabricObject) => void;
  remove: (obj: FabricObject) => void;
  getObjects: () => FabricObject[];
  getActiveObject: () => FabricObject | null;
  setActiveObject: (obj: FabricObject) => void;
  discardActiveObject: () => void;
  getWidth: () => number;
  getHeight: () => number;
  setWidth: (value: number) => void;
  setHeight: (value: number) => void;
  setZoom: (value: number) => void;
  setViewportTransform: (value: number[]) => void;
  viewportTransform: number[];
  getCenter: () => { left: number; top: number };
  zoomToPoint: (point: { x: number; y: number }, zoom: number) => void;
  bringForward: (obj: FabricObject) => void;
  sendBackwards: (obj: FabricObject) => void;
  requestRenderAll: () => void;
  renderAll: () => void;
  toDataURL: (opts?: { format?: string; quality?: number; multiplier?: number }) => string;
  toSVG: () => string;
  isDrawingMode?: boolean;
};

async function loadFabric(): Promise<FabricModule> {
  const mod = await import("fabric");
  return (mod.fabric ?? mod.default?.fabric ?? mod.default) as unknown as FabricModule;
}

function fitCanvasToContent(fabricCanvas: FabricCanvas) {
  const objects = fabricCanvas.getObjects();
  const maxW = Math.min(typeof window !== "undefined" ? window.innerWidth - 80 : 880, 1400);
  const maxH = Math.min(typeof window !== "undefined" ? window.innerHeight - 220 : 640, 900);

  if (!objects.length) {
    fabricCanvas.setWidth(Math.max(320, Math.min(880, maxW)));
    fabricCanvas.setHeight(Math.max(240, Math.min(640, maxH)));
    fabricCanvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
    fabricCanvas.setZoom(1);
    fabricCanvas.renderAll();
    return;
  }

  const bounds = objects.reduce<{
    left: number;
    top: number;
    right: number;
    bottom: number;
  } | null>((acc, obj) => {
    const rect = (obj as FabricObject & {
      getBoundingRect: (absolute?: boolean, calculate?: boolean) => {
        left: number;
        top: number;
        width: number;
        height: number;
      };
    }).getBoundingRect(true, true);

    if (!acc) {
      return {
        left: rect.left,
        top: rect.top,
        right: rect.left + rect.width,
        bottom: rect.top + rect.height,
      };
    }

    return {
      left: Math.min(acc.left, rect.left),
      top: Math.min(acc.top, rect.top),
      right: Math.max(acc.right, rect.left + rect.width),
      bottom: Math.max(acc.bottom, rect.top + rect.height),
    };
  }, null);

  if (!bounds) return;

  const pad = 32;
  const contentW = bounds.right - bounds.left;
  const contentH = bounds.bottom - bounds.top;
  const scale = Math.min(
    (maxW - pad * 2) / Math.max(contentW, 1),
    (maxH - pad * 2) / Math.max(contentH, 1),
    1
  );

  fabricCanvas.setWidth(Math.max(320, maxW));
  fabricCanvas.setHeight(Math.max(240, maxH));
  fabricCanvas.setZoom(1);
  fabricCanvas.setViewportTransform([1, 0, 0, 1, 0, 0]);

  const offsetX = (fabricCanvas.getWidth() - contentW * scale) / 2 - bounds.left * scale;
  const offsetY = (fabricCanvas.getHeight() - contentH * scale) / 2 - bounds.top * scale;
  fabricCanvas.setViewportTransform([scale, 0, 0, scale, offsetX, offsetY]);
  fabricCanvas.renderAll();
}

function promoteTextToIText(
  fabric: FabricModule,
  obj: FabricObject,
  callback: (ready: FabricObject) => void
) {
  if (!obj || obj.type !== "text") {
    callback(obj);
    return;
  }

  const targetWidth = obj._ocrTargetWidth ?? null;
  const iText = new fabric.IText(obj.text || "", {
    left: obj.left,
    top: obj.top,
    fontFamily: obj.fontFamily || "Arial, Helvetica, sans-serif",
    fontSize: obj.fontSize || 16,
    fontWeight: obj.fontWeight || "normal",
    fill: obj.fill || "#1a1d24",
    scaleX: obj.scaleX,
    scaleY: obj.scaleY,
    angle: obj.angle,
    originX: obj.originX,
    originY: obj.originY,
    selectable: true,
    evented: true,
  });

  if (targetWidth && targetWidth > 0 && iText.getScaledWidth) {
    const measuredWidth = iText.getScaledWidth();
    if (measuredWidth > 0) {
      iText.set("scaleX", (targetWidth / measuredWidth) * (obj.scaleX || 1));
    }
  }

  callback(iText);
}

async function loadSvgIntoCanvas(
  fabric: FabricModule,
  fabricCanvas: FabricCanvas,
  svgString: string
) {
  fabricCanvas.getObjects().forEach((obj) => fabricCanvas.remove(obj));

  await new Promise<void>((resolve, reject) => {
    fabric.loadSVGFromString(
      svgString,
      (objects) => {
        try {
          if (!objects.length) {
            resolve();
            return;
          }

          let pending = objects.length;
          const readyObjects: FabricObject[] = [];

          objects.forEach((obj) => {
            promoteTextToIText(fabric, obj, (ready) => {
              ready.set({ selectable: true, evented: true });
              readyObjects.push(ready);
              pending -= 1;

              if (pending === 0) {
                readyObjects.forEach((item) => fabricCanvas.add(item));
                fabricCanvas.renderAll();
                fitCanvasToContent(fabricCanvas);
                resolve();
              }
            });
          });
        } catch (err) {
          reject(err);
        }
      },
      (element, obj) => {
        if (element?.getAttribute && obj?.type === "text") {
          const textLengthAttr = element.getAttribute("textLength");
          if (textLengthAttr) {
            const targetWidth = parseFloat(textLengthAttr);
            if (!Number.isNaN(targetWidth) && targetWidth > 0) {
              obj._ocrTargetWidth = targetWidth;
            }
          }
        }
        obj.set({ selectable: true, evented: true });
      }
    );
  });
}

export function CanvasEditor({
  open,
  onOpenChange,
  image,
  onSaveToChat,
}: CanvasEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<FabricModule | null>(null);
  const fabricCanvasRef = useRef<FabricCanvas | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Vectorizing image…");
  const [fillColor, setFillColor] = useState("#6366f1");
  const [isReconstructing, setIsReconstructing] = useState(false);

  const destroyFabricCanvas = useCallback(() => {
    fabricCanvasRef.current?.dispose();
    fabricCanvasRef.current = null;
  }, []);

  const loadCanvas = useCallback(
    async (svgString: string, readyMessage: string) => {
      const fabric = fabricRef.current ?? (await loadFabric());
      fabricRef.current = fabric;

      const canvasEl = canvasRef.current;
      if (!canvasEl || !fabric?.Canvas) {
        throw new Error("Fabric.js failed to load");
      }

      destroyFabricCanvas();
      const canvas = new fabric.Canvas(canvasEl, {
        preserveObjectStacking: true,
        selection: true,
      });
      fabricCanvasRef.current = canvas;

      await loadSvgIntoCanvas(fabric, canvas, svgString);
      setStatusMessage(readyMessage);
    },
    [destroyFabricCanvas]
  );

  useEffect(() => {
    if (!open) return;

    let cancelled = false;

    async function init() {
      setLoading(true);
      setError(null);
      setStatusMessage("Vectorizing image…");

      try {
        const result = await api.vectorizeImage({
          filename: image.filename,
          image_data_url: image.imageDataUrl,
          prompt: image.prompt,
        });
        if (cancelled) return;

        await loadCanvas(
          result.svg,
          "Select shapes to move, recolor, or transform. Add text with Add text."
        );
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load canvas");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void init();

    return () => {
      cancelled = true;
      destroyFabricCanvas();
    };
  }, [open, image, loadCanvas, destroyFabricCanvas]);

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Delete" || event.key === "Backspace") {
        const tag = document.activeElement?.tagName ?? "";
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        const active = fabricCanvasRef.current?.getActiveObject();
        if (active?.isEditing) return;
        event.preventDefault();
        deleteSelection();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  const getActiveObject = () => fabricCanvasRef.current?.getActiveObject() ?? null;

  const applyFillColor = (color: string) => {
    const canvas = fabricCanvasRef.current;
    const obj = getActiveObject();
    if (!canvas || !obj) return;

    if (obj.type === "group" && obj.getObjects) {
      obj.getObjects().forEach((child) => {
        if ("fill" in child) child.set("fill", color);
        if (child.stroke) child.set("stroke", color);
      });
    } else {
      if ("fill" in obj) obj.set("fill", color);
      if (obj.stroke) obj.set("stroke", color);
    }

    canvas.requestRenderAll();
  };

  const addTextObject = () => {
    const fabric = fabricRef.current;
    const canvas = fabricCanvasRef.current;
    if (!fabric || !canvas) return;

    const text = new fabric.IText("Label", {
      left: canvas.getWidth() / 2 - 40,
      top: canvas.getHeight() / 2 - 12,
      fontFamily: "Arial, Helvetica, sans-serif",
      fontSize: 22,
      fill: fillColor,
    });

    canvas.add(text);
    canvas.setActiveObject(text);
    text.enterEditing?.();
    text.selectAll?.();
    canvas.requestRenderAll();
  };

  const deleteSelection = () => {
    const canvas = fabricCanvasRef.current;
    const obj = getActiveObject();
    if (!canvas || !obj) return;

    if (obj.type === "activeSelection" && obj.getObjects) {
      obj.getObjects().forEach((item) => canvas.remove(item));
    } else {
      canvas.remove(obj);
    }

    canvas.discardActiveObject();
    canvas.requestRenderAll();
  };

  const moveLayer = (direction: "forward" | "backward") => {
    const canvas = fabricCanvasRef.current;
    const obj = getActiveObject();
    if (!canvas || !obj) return;

    if (direction === "forward") {
      canvas.bringForward(obj);
    } else {
      canvas.sendBackwards(obj);
    }

    canvas.requestRenderAll();
  };

  const applyZoom = (factor: number) => {
    const fabric = fabricRef.current;
    const canvas = fabricCanvasRef.current;
    if (!fabric || !canvas) return;

    const vpt = canvas.viewportTransform.slice();
    let zoom = vpt[0] * factor;
    zoom = Math.max(0.05, Math.min(zoom, 8));
    const center = canvas.getCenter();
    canvas.zoomToPoint(new fabric.Point(center.left, center.top), zoom);
    canvas.requestRenderAll();
  };

  const exportSvg = () => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    const blob = new Blob([canvas.toSVG()], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${image.filename?.replace(".png", "") || "figure"}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportPngDataUrl = () => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return null;
    return canvas.toDataURL({ format: "png", quality: 1, multiplier: 3 });
  };

  const exportPng = () => {
    const dataUrl = exportPngDataUrl();
    if (!dataUrl) return;
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = `${image.filename || "figure"}.png`;
    link.click();
  };

  const saveToChat = () => {
    const dataUrl = exportPngDataUrl();
    if (!dataUrl || !onSaveToChat) return;
    onSaveToChat(dataUrl);
  };

  const runAiReconstruct = async () => {
    if (isReconstructing) return;

    setIsReconstructing(true);
    setLoading(true);
    setError(null);
    setStatusMessage("Reconstructing diagram with AI…");

    try {
      const result = await api.refineSvgCodegen({
        filename: image.filename,
        image_data_url: image.imageDataUrl,
      });

      await loadCanvas(
        result.svg,
        result.iterations
          ? `Reconstructed in ${result.iterations} iteration(s). Edit shapes or export when ready.`
          : "Reconstructed. Edit shapes or export when ready."
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "AI reconstruction failed");
      setStatusMessage("Select shapes to move, recolor, or transform.");
    } finally {
      setLoading(false);
      setIsReconstructing(false);
    }
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-5xl">
        <SheetHeader>
          <SheetTitle>Edit in Canvas</SheetTitle>
          <p className="text-sm text-muted-foreground">{statusMessage}</p>
        </SheetHeader>

        <div className="mt-4 flex min-h-0 flex-1 flex-col gap-4">
          {!loading && (
            <div className="flex flex-wrap items-end gap-2 rounded-lg border bg-muted/30 p-3">
              <Button size="sm" variant="outline" onClick={addTextObject}>
                <Type className="h-4 w-4" />
                Add text
              </Button>

              <div className="space-y-1">
                <Label htmlFor="canvas-fill-color" className="text-xs">
                  Color
                </Label>
                <Input
                  id="canvas-fill-color"
                  type="color"
                  value={fillColor}
                  className="h-9 w-14 cursor-pointer p-1"
                  onChange={(event) => {
                    setFillColor(event.target.value);
                    applyFillColor(event.target.value);
                  }}
                />
              </div>

              <Button size="sm" variant="outline" onClick={deleteSelection}>
                <Trash2 className="h-4 w-4" />
                Delete
              </Button>
              <Button size="sm" variant="outline" onClick={() => moveLayer("forward")}>
                <ArrowUp className="h-4 w-4" />
                Forward
              </Button>
              <Button size="sm" variant="outline" onClick={() => moveLayer("backward")}>
                <ArrowDown className="h-4 w-4" />
                Back
              </Button>

              <div className="mx-1 hidden h-8 w-px bg-border sm:block" />

              <Button size="sm" variant="outline" onClick={() => applyZoom(1 / 1.15)}>
                <Minus className="h-4 w-4" />
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  const canvas = fabricCanvasRef.current;
                  if (canvas) fitCanvasToContent(canvas);
                }}
              >
                Fit
              </Button>
              <Button size="sm" variant="outline" onClick={() => applyZoom(1.15)}>
                <Plus className="h-4 w-4" />
              </Button>

              <div className="mx-1 hidden h-8 w-px bg-border sm:block" />

              <Button size="sm" variant="outline" onClick={exportSvg}>
                <Download className="h-4 w-4" />
                SVG
              </Button>
              <Button size="sm" variant="outline" onClick={exportPng}>
                <Download className="h-4 w-4" />
                PNG
              </Button>
              {onSaveToChat && (
                <Button size="sm" variant="default" onClick={saveToChat}>
                  Save to chat
                </Button>
              )}
              <Button
                size="sm"
                variant="secondary"
                disabled={isReconstructing}
                onClick={() => void runAiReconstruct()}
              >
                <Sparkles className="h-4 w-4" />
                Reconstruct (AI)
              </Button>
            </div>
          )}

          {error && <Alert className="border-destructive/50 text-destructive">{error}</Alert>}

          {loading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              {statusMessage}
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-auto rounded-lg border bg-muted/30 p-2">
            <canvas ref={canvasRef} className="mx-auto max-w-full" />
          </div>

          <p className="truncate text-xs text-muted-foreground">
            Source: {resolveImageSrc(image).slice(0, 80)}…
          </p>
        </div>
      </SheetContent>
    </Sheet>
  );
}
