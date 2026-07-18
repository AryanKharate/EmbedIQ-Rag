/**
 * auth.ts
 * Manages JWT tokens in localStorage and provides auth utilities
 * used by the API client and route guard.
 */
import { jwtDecode } from "jwt-decode";

const ACCESS_KEY = "embediq_access";
const REFRESH_KEY = "embediq_refresh";
const USER_KEY = "embediq_user";

export interface AuthUser {
  id: number;
  email: string;
  display_name: string;
}

interface JwtPayload {
  exp: number;
  user_id: number;
}

// ─── Token storage ────────────────────────────────────────────────────────────

export function setTokens(access: string, refresh: string, user: AuthUser) {
  if (typeof window !== "undefined") {
    localStorage.setItem(ACCESS_KEY, access);
    localStorage.setItem(REFRESH_KEY, refresh);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }
}

export function clearTokens() {
  if (typeof window !== "undefined") {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem("rag.threads.v1");
  }
}

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_KEY);
}

export function getUser(): AuthUser | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
}

// ─── Auth checks ──────────────────────────────────────────────────────────────

/** Returns true if a non-expired access token exists in localStorage. */
export function isAuthenticated(): boolean {
  const token = getAccessToken();
  if (!token) return false;
  try {
    const { exp } = jwtDecode<JwtPayload>(token);
    // Consider token expired 30s before actual expiry to avoid edge cases
    return Date.now() / 1000 < exp - 30;
  } catch {
    return false;
  }
}

/** Attempt to refresh the access token using the stored refresh token.
 *  Returns the new access token on success, null on failure.
 */
export async function refreshAccessToken(): Promise<string | null> {
  const refresh = getRefreshToken();
  if (!refresh) return null;
  try {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh }),
    });
    if (!res.ok) {
      clearTokens();
      return null;
    }
    const data = (await res.json()) as { access: string };
    if (typeof window !== "undefined") {
      localStorage.setItem(ACCESS_KEY, data.access);
    }
    return data.access;
  } catch {
    clearTokens();
    return null;
  }
}

/** Redirect to login page, clearing tokens first. */
export function logout() {
  clearTokens();
  if (typeof window !== "undefined") {
    window.location.href = "/login";
  }
}
