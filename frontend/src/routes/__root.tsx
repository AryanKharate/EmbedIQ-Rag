import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  useNavigate,
  HeadContent,
  Scripts,
  useRouterState,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";
import { GoogleOAuthProvider } from "@react-oauth/google";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";
import { AppSidebar } from "@/components/app-sidebar";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Toaster } from "@/components/ui/sonner";
import { isAuthenticated, logout } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { LogOut } from "lucide-react";
import { useHydrated } from "@/hooks/use-hydrated";

// Google OAuth client ID — set VITE_GOOGLE_CLIENT_ID in your .env
// May be empty string if not configured; we conditionally wrap with the provider.
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";
const hasGoogleAuth =
  !!GOOGLE_CLIENT_ID && !GOOGLE_CLIENT_ID.startsWith("your-google-client-id");

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-bold text-foreground">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">
          Page not found
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Go home
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportLovableError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong on our end. You can try refreshing or head back
          home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()(
  {
    head: () => ({
      meta: [
        { charSet: "utf-8" },
        { name: "viewport", content: "width=device-width, initial-scale=1" },
        { title: "EmbedIQ — Chat with your documents" },
        {
          name: "description",
          content:
            "EmbedIQ: An enterprise-grade RAG platform. Chat with your documents using HyDE, CRAG, and vector search.",
        },
        { name: "author", content: "EmbedIQ" },
        { property: "og:title", content: "EmbedIQ — Chat with your documents" },
        {
          property: "og:description",
          content:
            "EmbedIQ: An enterprise-grade RAG platform. Chat with your documents using HyDE, CRAG, and vector search.",
        },
        { property: "og:type", content: "website" },
        { name: "twitter:card", content: "summary_large_image" },
      ],
      links: [
        { rel: "stylesheet", href: appCss },
        { rel: "icon", href: "/favicon.ico", type: "image/x-icon" },
      ],
    }),
    shellComponent: RootShell,
    component: RootComponent,
    notFoundComponent: NotFoundComponent,
    errorComponent: ErrorComponent,
  },
);

function RootShell({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

/**
 * Conditionally wraps children with GoogleOAuthProvider only when a real
 * client ID is configured. This prevents the provider from crashing when
 * VITE_GOOGLE_CLIENT_ID is empty or a placeholder.
 */
function MaybeGoogleProvider({ children }: { children: ReactNode }) {
  if (!hasGoogleAuth) return <>{children}</>;
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      {children}
    </GoogleOAuthProvider>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  const routerState = useRouterState();
  const currentPath = routerState.location.pathname;
  const isPublicRoute = currentPath === "/login" || currentPath === "/register";

  // useHydrated() is false on the server; true only after browser mounts.
  // This prevents any localStorage/window access during SSR.
  const hydrated = useHydrated();
  const navigate = useNavigate();

  useEffect(() => {
    if (!hydrated) return;
    if (!isPublicRoute && !isAuthenticated()) {
      void navigate({ to: "/login" });
    }
  }, [hydrated, isPublicRoute, navigate]);

  // Before hydration: render the outlet with no providers that touch browser APIs
  if (!hydrated) {
    return (
      <QueryClientProvider client={queryClient}>
        <Outlet />
        <Toaster />
      </QueryClientProvider>
    );
  }

  // After hydration: safe to check localStorage / render browser-only providers
  return (
    <MaybeGoogleProvider>
      <QueryClientProvider client={queryClient}>
        {isPublicRoute ? (
          // Public pages (login/register) — no sidebar
          <>
            <Outlet />
            <Toaster />
          </>
        ) : !isAuthenticated() ? (
          // Not authenticated — render nothing while useEffect redirect fires.
          // This prevents AppSidebar from mounting and calling the API unauthenticated.
          <Toaster />
        ) : (
          // Authenticated — full app shell with sidebar
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset className="flex h-svh flex-col">
              <header className="flex h-12 shrink-0 items-center gap-2 border-b px-3">
                <SidebarTrigger />
                <span className="text-sm font-medium text-muted-foreground">
                  EmbedIQ
                </span>
                <div className="ml-auto">
                  <Button
                    id="logout-btn"
                    variant="ghost"
                    size="sm"
                    className="gap-1.5 text-muted-foreground hover:text-foreground"
                    onClick={() => logout()}
                  >
                    <LogOut className="h-4 w-4" />
                    <span className="hidden sm:inline">Sign out</span>
                  </Button>
                </div>
              </header>
              <div className="min-h-0 flex-1">
                <Outlet />
              </div>
            </SidebarInset>
            <Toaster />
          </SidebarProvider>
        )}
      </QueryClientProvider>
    </MaybeGoogleProvider>
  );
}
