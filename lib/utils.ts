import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function generateId(prefix = "id") {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export function getClientSessionId() {
  if (typeof window === "undefined") return "";
  const key = "client_session_id";
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = generateId("session");
    sessionStorage.setItem(key, id);
  }
  return id;
}
