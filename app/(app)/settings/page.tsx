"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { useGenerationStore } from "@/lib/store/generation-store";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const { model, setModel, aspectRatio, setAspectRatio } = useGenerationStore();
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    void api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <AppShell>
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-8 md:px-6">
        <div>
          <h2 className="text-2xl font-semibold">Settings</h2>
          <p className="text-sm text-muted-foreground">
            Configure defaults and view system status
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Appearance</CardTitle>
            <CardDescription>Theme and display preferences</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Theme</Label>
              <Select
                value={mounted ? (theme ?? "dark") : "dark"}
                onValueChange={setTheme}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="dark">Dark</SelectItem>
                  <SelectItem value="light">Light</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generation defaults</CardTitle>
            <CardDescription>Default model and output settings for Studio</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Image model</Label>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gemini-3-pro-image-preview">Gemini 3 Pro Image</SelectItem>
                  <SelectItem value="gemini-3.1-flash-image-preview">Gemini 3.1 Flash Image</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Default aspect ratio</Label>
              <Select value={aspectRatio} onValueChange={(v) => setAspectRatio(v as typeof aspectRatio)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["auto", "1:1", "16:9", "9:16", "4:3", "3:4"].map((ratio) => (
                    <SelectItem key={ratio} value={ratio}>
                      {ratio}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System status</CardTitle>
            <CardDescription>Backend integration health</CardDescription>
          </CardHeader>
          <CardContent>
            {!health ? (
              <Alert>Unable to reach the API. Restart the dev server and confirm GOOGLE_GENERATIVE_AI_API_KEY is set in .env.</Alert>
            ) : (
              <div className="flex flex-wrap gap-2">
                {Object.entries(health).map(([key, value]) => (
                  <Badge key={key} variant={value ? "default" : "secondary"}>
                    {key}: {String(value)}
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Coming soon</CardTitle>
            <CardDescription>Planned platform capabilities</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {["Clerk Auth", "Vercel Blob Storage", "Stripe Subscriptions", "Team Workspaces", "Real-time Jobs"].map(
              (item) => (
                <Badge key={item} variant="outline">
                  {item}
                </Badge>
              )
            )}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  );
}
