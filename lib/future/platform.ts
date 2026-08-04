/**
 * Future platform integrations — stubs for upcoming SaaS features.
 * These modules will be wired when auth, billing, and persistent storage ship.
 */

export type WorkspacePlan = "free" | "pro" | "team";

export interface Workspace {
  id: string;
  name: string;
  plan: WorkspacePlan;
}

export interface FutureAuthContext {
  userId: string | null;
  workspaceId: string | null;
}

export const futureAuth: FutureAuthContext = {
  userId: null,
  workspaceId: null,
};

export const futureFeatures = {
  clerkAuth: false,
  vercelBlob: false,
  stripeSubscriptions: false,
  teamWorkspaces: false,
  realtimeJobs: false,
} as const;
