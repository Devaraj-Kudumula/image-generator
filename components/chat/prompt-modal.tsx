"use client";

import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface PromptModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  body: string;
}

export function PromptModal({ open, onOpenChange, title, body }: PromptModalProps) {
  const copyBody = () => {
    if (navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(body);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-2xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <div className="max-h-[60vh] overflow-auto rounded-md border bg-muted/30 p-4">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
            {body}
          </pre>
        </div>
        <div className="flex justify-end">
          <Button variant="outline" size="sm" onClick={copyBody}>
            <Copy className="h-4 w-4" />
            Copy
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
