import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

import { TooltipProvider } from "@/components/ui/tooltip";
import { OpenInDesktopButton } from "./OpenInDesktopButton";

/** The header mounts inside a TooltipProvider; mirror that here. */
function renderButton(conversationId = "conv_abc") {
  return render(
    <TooltipProvider>
      <OpenInDesktopButton conversationId={conversationId} />
    </TooltipProvider>,
  );
}

// `useRebasePath` is the standalone identity rebase in these tests; the
// embedded (basenamed) case is already pinned by PermissionsModal's suite,
// which exercises the same `getDeepLink`.
vi.mock("@/lib/routing", () => ({
  useRebasePath: () => (path: string) => path,
}));

/**
 * jsdom's `window.location` is non-configurable, so swap in a plain object to
 * control the origin/host `getDeepLink` reads, and capture `href` writes —
 * assigning a custom scheme is the whole mechanism under test.
 */
function stubLocation(origin: string): { hrefWrites: string[]; restore: () => void } {
  const original = window.location;
  const hrefWrites: string[] = [];
  const { host } = new URL(origin);
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      ...original,
      origin,
      host,
      set href(value: string) {
        hrefWrites.push(value);
      },
      get href() {
        return origin;
      },
    },
  });
  return {
    hrefWrites,
    restore: () =>
      Object.defineProperty(window, "location", { configurable: true, value: original }),
  };
}

function stubVisibility(state: DocumentVisibilityState): void {
  Object.defineProperty(document, "visibilityState", { configurable: true, value: state });
}

const visibilityDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, "visibilityState");

let location: ReturnType<typeof stubLocation>;

beforeEach(() => {
  vi.useFakeTimers();
  location = stubLocation("https://app.example.com");
});

afterEach(() => {
  vi.useRealTimers();
  location.restore();
  delete (window as { omnigentDesktop?: unknown }).omnigentDesktop;
  if (visibilityDescriptor) {
    Object.defineProperty(Document.prototype, "visibilityState", visibilityDescriptor);
  }
});

describe("OpenInDesktopButton", () => {
  it("navigates to the session's omnigent:// deep link on click", () => {
    renderButton();

    fireEvent.click(screen.getByTestId("open-in-desktop"));

    expect(location.hrefWrites).toEqual(["omnigent://app.example.com/c/conv_abc"]);
  });

  it("keeps the port in the host so loopback servers resolve", () => {
    location.restore();
    location = stubLocation("http://localhost:8000");
    renderButton();

    fireEvent.click(screen.getByTestId("open-in-desktop"));

    expect(location.hrefWrites).toEqual(["omnigent://localhost:8000/c/conv_abc"]);
  });

  it("renders nothing inside the desktop shell, where the session is already open", () => {
    (window as { omnigentDesktop?: unknown }).omnigentDesktop = { kind: "electron" };

    renderButton();

    expect(screen.queryByTestId("open-in-desktop")).not.toBeInTheDocument();
  });

  it("offers the download once the link goes unhandled and the page is still visible", () => {
    stubVisibility("visible");
    renderButton();

    fireEvent.click(screen.getByTestId("open-in-desktop"));
    expect(screen.queryByTestId("get-desktop-app")).not.toBeInTheDocument();

    act(() => void vi.runAllTimers());

    const fallback = screen.getByTestId("get-desktop-app");
    expect(fallback).toHaveAttribute("href", "https://omnigent.ai/download/mac");
  });

  it("stays quiet when the hand-off succeeds and backgrounds the page", () => {
    // A successful open pulls OS focus to the desktop app, hiding this page —
    // prompting a download there would contradict what the user just saw.
    stubVisibility("hidden");
    renderButton();

    fireEvent.click(screen.getByTestId("open-in-desktop"));
    act(() => void vi.runAllTimers());

    expect(screen.queryByTestId("get-desktop-app")).not.toBeInTheDocument();
    expect(screen.getByTestId("open-in-desktop")).toBeInTheDocument();
  });
});
