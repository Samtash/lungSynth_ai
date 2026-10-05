import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { ShieldCheck, Lock, MonitorSmartphone, Loader2, TriangleAlert } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Logo } from "@/components/Logo";
import { setUser } from "@/lib/lungsynth";
import {
  GOOGLE_CLIENT_ID,
  loadGoogleIdentity,
  signInWithGoogle,
  type GoogleCredentialResponse,
} from "@/lib/google-auth";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — LungSynth AI" },
      { name: "description", content: "Secure Google sign-in for radiologists and researchers using LungSynth AI 4D CT reconstruction." },
      { property: "og:title", content: "Sign in — LungSynth AI" },
      { property: "og:description", content: "Sign in with Google to use the LungSynth AI 4D CT research demo." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Login,
});

const assurances = [
  { icon: ShieldCheck, label: "Secure Authentication" },
  { icon: Lock, label: "Your data stays local" },
  { icon: MonitorSmartphone, label: "Responsive design" },
];

type Status = "loading" | "ready" | "exchanging" | "unavailable";

function Login() {
  const navigate = useNavigate();
  const buttonSlot = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);

  // Called by GIS once the user picks an account. `credential` is the
  // signed ID token; the backend verifies it, we never trust it here.
  const handleCredential = useCallback(
    (response: GoogleCredentialResponse) => {
      setError(null);
      setStatus("exchanging");
      signInWithGoogle(response.credential)
        .then((user) => {
          setUser(user);
          navigate({ to: "/dashboard" });
        })
        .catch((err: Error) => {
          setError(err.message || "Sign-in failed. Please try again.");
          setStatus("ready");
        });
    },
    [navigate],
  );

  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) {
      setStatus("unavailable");
      setError("VITE_GOOGLE_CLIENT_ID is not set. Add it to your .env and restart the dev server.");
      return;
    }

    let cancelled = false;

    loadGoogleIdentity()
      .then((google) => {
        if (cancelled || !buttonSlot.current) return;

        google.accounts.id.initialize({
          client_id: GOOGLE_CLIENT_ID,
          callback: handleCredential,
          ux_mode: "popup",
          auto_select: false,
          cancel_on_tap_outside: true,
        });

        // Google only accepts a pixel width between 200 and 400.
        const slotWidth = buttonSlot.current.offsetWidth || 320;
        google.accounts.id.renderButton(buttonSlot.current, {
          theme: "outline",
          size: "large",
          text: "continue_with",
          shape: "rectangular",
          logo_alignment: "left",
          width: Math.min(400, Math.max(200, Math.round(slotWidth))),
        });

        setStatus("ready");
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setStatus("unavailable");
        setError(err.message);
      });

    return () => {
      cancelled = true;
    };
  }, [handleCredential]);

  return (
    <div className="grid-medical flex min-h-screen items-center justify-center bg-background px-4 py-12">
      <div className="animate-fade-up w-full max-w-md">
        <div className="rounded-3xl border border-border bg-card p-8 shadow-[var(--shadow-lift)] sm:p-10">
          <div className="flex flex-col items-center text-center">
            <div className="text-primary">
              <Logo size={64} />
            </div>
            <h1 className="mt-6 text-2xl font-extrabold tracking-tight sm:text-[26px]">
              Welcome to LungSynth AI
            </h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Generate high-quality intermediate lung CT phases using AI diffusion models.
            </p>
          </div>

          <div className="mt-8 min-h-12">
            {/* GIS draws its own button into this element. */}
            <div ref={buttonSlot} className="flex justify-center [&>div]:w-full" />

            {status === "loading" && (
              <div className="flex h-12 items-center justify-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Loading Google Sign-In…
              </div>
            )}

            {status === "exchanging" && (
              <div className="mt-3 flex items-center justify-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Verifying your account…
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-destructive/30 bg-destructive/5 p-3 text-xs leading-relaxed text-destructive">
              <TriangleAlert className="mt-px size-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="mt-8 flex flex-col gap-3 border-t border-border pt-6">
            {assurances.map((a) => (
              <div key={a.label} className="flex items-center gap-2.5 text-xs font-medium text-muted-foreground">
                <a.icon className="size-4 text-secondary" />
                {a.label}
              </div>
            ))}
          </div>
        </div>
        <p className="mt-6 text-center text-xs text-muted-foreground">
          Access restricted to authorised clinical and research personnel.
        </p>
      </div>
    </div>
  );
}
