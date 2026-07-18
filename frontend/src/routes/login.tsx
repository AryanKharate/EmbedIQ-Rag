import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useState, useEffect, type FormEvent } from "react";
import { Sparkles, Mail, Lock, Eye, EyeOff, Chrome } from "lucide-react";
import { authApi } from "@/lib/api";
import { setTokens, isAuthenticated } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useHydrated } from "@/hooks/use-hydrated";

export const Route = createFileRoute("/login")({
  component: LoginPage,
});

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";
const hasGoogleAuth =
  !!GOOGLE_CLIENT_ID && !GOOGLE_CLIENT_ID.startsWith("your-google-client-id");

import { useGoogleLogin } from "@react-oauth/google";

/**
 * Separate component so useGoogleLogin hook is only called when
 * GoogleOAuthProvider is actually in the tree (i.e. hasGoogleAuth = true).
 * Calling useGoogleLogin outside a provider crashes the app.
 */
function GoogleSignInButton({
  disabled,
  onError,
  onSuccess,
}: {
  disabled: boolean;
  onError: (msg: string) => void;
  onSuccess: (accessToken: string) => void;
}) {
  const googleLogin = useGoogleLogin({
    onSuccess: (res: { access_token: string }) => onSuccess(res.access_token),
    onError: () => onError("Google sign-in was cancelled or failed."),
  });
  return (
    <Button
      id="google-signin-btn"
      type="button"
      variant="outline"
      className="w-full gap-2"
      disabled={disabled}
      onClick={() => googleLogin()}
    >
      <Chrome className="h-4 w-4" />
      Continue with Google
    </Button>
  );
}

function LoginPage() {
  const navigate = useNavigate();
  const hydrated = useHydrated();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Redirect after hydration if already logged in
  useEffect(() => {
    if (hydrated && isAuthenticated()) {
      void navigate({ to: "/" });
    }
  }, [hydrated, navigate]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await authApi.login(email.trim().toLowerCase(), password);
      setTokens(res.access, res.refresh, res.user);
      void navigate({ to: "/" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleGoogleSuccess(accessToken: string) {
    setError(null);
    setLoading(true);
    try {
      const res = await authApi.googleAuth(accessToken);
      setTokens(res.access, res.refresh, res.user);
      void navigate({ to: "/" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Google sign-in failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      {/* Background gradient orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-primary/5 blur-3xl" />
      </div>

      <div className="relative w-full max-w-sm">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg shadow-primary/25">
            <Sparkles className="h-6 w-6" />
          </div>
          <h1 className="mt-4 text-2xl font-semibold tracking-tight text-foreground">
            Welcome back
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Sign in to your EmbedIQ account
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border bg-card/60 p-6 shadow-xl backdrop-blur">
          {/* Google button — only rendered when provider + client ID are available */}
          {hasGoogleAuth && (
            <>
              <GoogleSignInButton
                disabled={loading}
                onError={setError}
                onSuccess={handleGoogleSuccess}
              />
              <div className="my-5 flex items-center gap-3">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs text-muted-foreground">or</span>
                <div className="h-px flex-1 bg-border" />
              </div>
            </>
          )}

          {/* Email/Password form */}
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="login-email">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="login-email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  className="pl-9"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="login-password">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="login-password"
                  type={showPw ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder="••••••••"
                  className="pl-9 pr-9"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  onClick={() => setShowPw((v) => !v)}
                  aria-label={showPw ? "Hide password" : "Show password"}
                >
                  {showPw ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            {error && (
              <p
                id="login-error"
                className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive"
              >
                {error}
              </p>
            )}

            <Button
              id="login-submit-btn"
              type="submit"
              className="w-full"
              disabled={loading || !email || !password}
            >
              {loading ? "Signing in…" : "Sign in"}
            </Button>
          </form>
        </div>

        {/* Footer */}
        <p className="mt-5 text-center text-sm text-muted-foreground">
          Don&apos;t have an account?{" "}
          <Link
            to="/register"
            className={cn(
              "font-medium text-primary underline-offset-4 hover:underline",
            )}
          >
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}
