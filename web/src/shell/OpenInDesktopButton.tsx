import { useCallback, useEffect, useRef, useState } from "react";
import { MonitorUpIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { DESKTOP_DOWNLOAD_URL, getDeepLink } from "@/lib/deepLink";
import { isElectronShell } from "@/lib/nativeBridge";
import { useRebasePath } from "@/lib/routing";
import { showToast } from "@/components/ui/toast";

/**
 * How long to wait after firing the link before offering the download.
 *
 * A custom scheme can't be feature-detected: if no app claims `omnigent://`,
 * navigation silently does nothing. So we wait, and if the page is still
 * here and visible, assume nothing handled it. Long enough that a successful
 * hand-off (which backgrounds the tab) wins the race on a slow machine.
 */
const FALLBACK_PROMPT_MS = 1500;

/**
 * Hand this session off to the desktop app via its `omnigent://` deep link.
 *
 * Shared by the desktop header button and the mobile session menu so the two
 * entry points can't drift apart. `unavailable` is true once we've concluded
 * no app claimed the link, which swaps the affordance over to "Get the
 * desktop app" rather than letting a dead button sit there.
 */
export function useOpenInDesktop(conversationId: string): {
  unavailable: boolean;
  openInDesktop: () => void;
} {
  const rebasePath = useRebasePath();
  const [unavailable, setUnavailable] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    };
  }, []);

  const openInDesktop = useCallback(() => {
    const link = getDeepLink(conversationId, rebasePath);
    // Assigning location is what actually triggers the OS handler. It never
    // navigates the page away: an unregistered scheme is a no-op, and a
    // registered one is intercepted by the OS.
    try {
      window.location.href = link;
    } catch (err) {
      console.warn("Failed to open the desktop deep link", err);
    }

    if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    timeoutRef.current = window.setTimeout(() => {
      // A successful hand-off pulls focus to the desktop app, so a hidden
      // page means it worked and we must stay quiet.
      if (document.visibilityState === "hidden") return;
      setUnavailable(true);
      showToast(
        <span className="flex min-w-0 flex-col gap-0.5">
          <span>Couldn't open the desktop app</span>
          <a
            href={DESKTOP_DOWNLOAD_URL}
            target="_blank"
            rel="noreferrer"
            className="truncate text-xs underline"
          >
            Download Omnigent for desktop
          </a>
        </span>,
        { duration: 8000 },
      );
    }, FALLBACK_PROMPT_MS);
  }, [conversationId, rebasePath]);

  return { unavailable, openInDesktop };
}

/**
 * Header action that opens this session in the desktop app.
 *
 * Renders nothing inside the desktop shell itself — the session is already
 * open there, so the button would be a no-op pointing at the current window.
 */
export function OpenInDesktopButton({ conversationId }: { conversationId: string }) {
  const { unavailable, openInDesktop } = useOpenInDesktop(conversationId);
  if (isElectronShell()) return null;

  const label = unavailable ? "Get the desktop app" : "Open in Desktop";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {unavailable ? (
          <Button
            asChild
            variant="ghost"
            size="icon"
            aria-label={label}
            data-testid="get-desktop-app"
            className="text-muted-foreground hover:text-foreground"
          >
            <a href={DESKTOP_DOWNLOAD_URL} target="_blank" rel="noreferrer">
              <MonitorUpIcon className="size-4" />
            </a>
          </Button>
        ) : (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={label}
            data-testid="open-in-desktop"
            onClick={openInDesktop}
            className="text-muted-foreground hover:text-foreground"
          >
            <MonitorUpIcon className="size-4" />
          </Button>
        )}
      </TooltipTrigger>
      {/* Bottom placement matches the header's other tooltips, keeping
          clear of the Electron shell's title-bar strip. */}
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  );
}

/**
 * Mobile three-dot menu counterpart of {@link OpenInDesktopButton}.
 *
 * Kept on phones because the same `omnigent://` link the desktop shell
 * claims is also claimed by the iOS app, so this opens the session natively
 * on the device the user is already holding.
 */
export function OpenInDesktopMenuItem({ conversationId }: { conversationId: string }) {
  const { openInDesktop } = useOpenInDesktop(conversationId);
  if (isElectronShell()) return null;
  return (
    <DropdownMenuItem
      onSelect={openInDesktop}
      data-testid="mobile-open-in-desktop"
      className="gap-2.5 px-2.5 py-2 text-base"
    >
      <MonitorUpIcon className="size-4" />
      Open in app
    </DropdownMenuItem>
  );
}
