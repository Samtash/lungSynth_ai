/**
 * Google Sign-In (Google Identity Services) helpers.
 *
 * Loads the GIS client script on demand, then exchanges the ID token it
 * returns for a user record from the backend (`POST /api/auth/google`,
 * implemented in backend/app/routers/auth.py).
 *
 * The script is loaded lazily inside an effect rather than in the route
 * head, because this app server renders and `document` does not exist
 * during SSR.
 */

import { API_BASE_URL } from "@/lib/api";
import type { StoredUser } from "@/lib/lungsynth";

export const GOOGLE_CLIENT_ID =
  (import.meta as unknown as { env?: Record<string, string> }).env?.["VITE_GOOGLE_CLIENT_ID"] ?? "";

const GIS_SRC = "https://accounts.google.com/gsi/client";

export type GoogleCredentialResponse = { credential: string };

type GoogleButtonOptions = {
  theme?: "outline" | "filled_blue" | "filled_black";
  size?: "small" | "medium" | "large";
  text?: "signin_with" | "signup_with" | "continue_with" | "signin";
  shape?: "rectangular" | "pill" | "circle" | "square";
  logo_alignment?: "left" | "center";
  width?: number;
};

type GoogleIdentity = {
  accounts: {
    id: {
      initialize(config: {
        client_id: string;
        callback: (response: GoogleCredentialResponse) => void;
        auto_select?: boolean;
        cancel_on_tap_outside?: boolean;
        ux_mode?: "popup" | "redirect";
      }): void;
      renderButton(parent: HTMLElement, options: GoogleButtonOptions): void;
      disableAutoSelect(): void;
    };
  };
};

declare global {
  interface Window {
    google?: GoogleIdentity;
  }
}

let loaderPromise: Promise<GoogleIdentity> | null = null;

/** Injects the GIS script once and resolves when `window.google` is usable. */
export function loadGoogleIdentity(): Promise<GoogleIdentity> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Google Sign-In can only load in the browser"));
  }
  if (window.google?.accounts?.id) return Promise.resolve(window.google);
  if (loaderPromise) return loaderPromise;

  loaderPromise = new Promise<GoogleIdentity>((resolve, reject) => {
    const done = () => {
      if (window.google?.accounts?.id) resolve(window.google);
      else reject(new Error("Google Identity Services loaded but is unavailable"));
    };
    const fail = () => {
      loaderPromise = null;
      reject(new Error("Could not reach accounts.google.com"));
    };

    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GIS_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", done);
      existing.addEventListener("error", fail);
      return;
    }

    const script = document.createElement("script");
    script.src = GIS_SRC;
    script.async = true;
    script.defer = true;
    script.onload = done;
    script.onerror = fail;
    document.head.appendChild(script);
  });

  return loaderPromise;
}

/**
 * Sends the Google ID token to the backend, which verifies its signature
 * against LUNGSYNTH_GOOGLE_CLIENT_ID and returns the user profile.
 */
export async function signInWithGoogle(credential: string): Promise<StoredUser> {
  const res = await fetch(`${API_BASE_URL}/api/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ credential }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // keep statusText
    }
    throw new Error(detail);
  }

  return (await res.json()) as StoredUser;
}

/** Stops Google from silently re-signing the user in after they log out. */
export function forgetGoogleSession() {
  try {
    window.google?.accounts.id.disableAutoSelect();
  } catch {
    // GIS never loaded, nothing to forget
  }
}
