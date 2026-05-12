/**
 * ShallWeSwim Debug Utility
 *
 * This script helps diagnose JavaScript and API loading issues across browsers.
 * It logs important events and can be activated by users with issues.
 */

// Check if debugging is enabled via URL parameter
function isDebugEnabled() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("debug") === "1";
}

const debugEnabled = isDebugEnabled();

// Debug state survives accidental duplicate script loads.
const debugState = window.SWS_DEBUG_STATE || {
  logs: [],
  browserInfo: {
    // Basic browser info
    userAgent: navigator.userAgent,
    language: navigator.language,
    cookiesEnabled: navigator.cookieEnabled,
    onLine: navigator.onLine,
    doNotTrack: navigator.doNotTrack,
    windowDimensions: `${window.innerWidth}x${window.innerHeight}`,
    devicePixelRatio: window.devicePixelRatio,
    timestamp: new Date().toISOString(),
  },
  apiCalls: [],
  domState: {},
  errors: [],
};
window.SWS_DEBUG_STATE = debugState;

// Store original browser APIs so wrappers cannot stack.
const originalConsoleError = console.error.swsOriginal || console.error;
const originalConsoleLog = console.log;
const originalFetch = window.fetch.swsOriginal || window.fetch;

// Add a log entry with timestamp
function logDebug(category, message, data = null) {
  const entry = {
    timestamp: new Date().toISOString(),
    category,
    message,
    data: data ? JSON.stringify(data).substring(0, 500) : null, // Truncate large data
  };
  debugState.logs.push(entry);
  if (debugEnabled) {
    // Use original console.log to avoid recursion if console.log is also patched
    originalConsoleLog.apply(console, [
      `[DEBUG:${category}] ${message}`,
      data || "",
    ]);
  }

  // Update UI if the debug panel is visible
  updateDebugPanel();
}

// Log an error
function logError(message, error) {
  const entry = {
    timestamp: new Date().toISOString(),
    message,
    errorName: error?.name,
    errorMessage: error?.message,
    stack: error?.stack,
  };
  debugState.errors.push(entry);

  // Update UI if the debug panel is visible
  updateDebugPanel();
}

// Track API calls
function trackApiCall(url, success, data, error) {
  const entry = {
    timestamp: new Date().toISOString(),
    url,
    success,
    data: success ? JSON.stringify(data).substring(0, 300) : null, // Truncate large data
    error: !success
      ? {
          message: error?.message,
          name: error?.name,
        }
      : null,
  };
  debugState.apiCalls.push(entry);

  // Update UI if the debug panel is visible
  updateDebugPanel();
}

// Capture current DOM state of important elements
function captureDomState() {
  const elements = {
    "water-temp": document.getElementById("water-temp")?.textContent,
    "past-tide-time": document.getElementById("past-tide-time")?.textContent,
    "next-tide-0-time":
      document.getElementById("next-tide-0-time")?.textContent,
    SWIMCONFIG: window.SWIMCONFIG ? "defined" : "undefined",
    locationCode: window.SWIMCONFIG?.locationCode,
  };

  debugState.domState = elements;
  logDebug("DOM", "DOM state captured", elements);
}

// Create and show debug panel
function showDebugPanel() {
  let panel = document.getElementById("sws-debug-panel");

  if (!panel) {
    panel = document.createElement("div");
    panel.id = "sws-debug-panel";
    panel.style.cssText = `
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      height: 300px;
      background: rgba(0, 0, 0, 0.85);
      color: #00ff00;
      font-family: monospace;
      z-index: 9999;
      overflow: auto;
      padding: 10px;
      font-size: 12px;
      border-top: 2px solid #00ff00;
    `;

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    closeBtn.style.cssText = `
      position: absolute;
      top: 5px;
      right: 5px;
      background: #333;
      color: white;
      border: 1px solid #00ff00;
      cursor: pointer;
      padding: 3px 8px;
    `;
    closeBtn.onclick = () => {
      panel.style.display = "none";
    };

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy Debug Data";
    copyBtn.style.cssText = `
      position: absolute;
      top: 5px;
      right: 60px;
      background: #333;
      color: white;
      border: 1px solid #00ff00;
      cursor: pointer;
      padding: 3px 8px;
    `;
    copyBtn.onclick = () => {
      const debugText = JSON.stringify(debugState, null, 2);
      navigator.clipboard
        .writeText(debugText)
        .then(() =>
          alert(
            "Debug info copied to clipboard. Please share with the site admin.",
          ),
        )
        .catch((err) => {
          console.error("Failed to copy:", err);
          alert(
            "Could not copy to clipboard. Please manually copy the data shown in the panel.",
          );
        });
    };

    const emailBtn = document.createElement("button");
    emailBtn.textContent = "Email Debug Data";
    emailBtn.style.cssText = `
      position: absolute;
      top: 5px;
      right: 180px;
      background: #333;
      color: white;
      border: 1px solid #00ff00;
      cursor: pointer;
      padding: 3px 8px;
    `;
    emailBtn.onclick = () => {
      const debugText = JSON.stringify(debugState, null, 2);
      const subject = "ShallWeSwim Debug Info";
      const body = `Browser: ${navigator.userAgent}\n\nDebug Data:\n${debugText}`;
      window.location.href = `mailto:admin@shallweswim.today?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    };

    const content = document.createElement("div");
    content.id = "sws-debug-content";

    panel.appendChild(closeBtn);
    panel.appendChild(copyBtn);
    panel.appendChild(emailBtn);
    panel.appendChild(content);

    document.body.appendChild(panel);
  } else {
    panel.style.display = "block";
  }

  updateDebugPanel();
}

function updateDebugPanel() {
  const panel = document.getElementById("sws-debug-content");
  if (!panel) return;

  panel.replaceChildren();

  appendHeading(panel, `Debug Info (${new Date().toLocaleTimeString()})`, 3);

  appendHeading(panel, "Browser Information", 4);
  appendParagraph(panel, `User Agent: ${debugState.browserInfo.userAgent}`);
  appendParagraph(
    panel,
    `Window Size: ${debugState.browserInfo.windowDimensions}`,
  );
  appendParagraph(panel, `Online: ${debugState.browserInfo.onLine}`);

  appendHeading(panel, "DOM State", 4);
  appendPreformatted(panel, JSON.stringify(debugState.domState, null, 2));

  if (debugState.errors.length > 0) {
    appendHeading(panel, "Recent Errors", 4);
    const recentErrors = debugState.errors.slice(-3);
    recentErrors.forEach((err) => {
      appendStatusLine(
        panel,
        `${err.timestamp}: ${err.message} - ${err.errorName}: ${err.errorMessage}`,
        false,
      );
    });
  }

  if (debugState.apiCalls.length > 0) {
    appendHeading(panel, "Recent API Calls", 4);
    const recentCalls = debugState.apiCalls.slice(-3);
    recentCalls.forEach((call) => {
      appendStatusLine(
        panel,
        `${call.timestamp}: ${call.url} - ${call.success ? "SUCCESS" : "FAILED"}`,
        call.success,
      );
    });
  }

  if (debugState.logs.length > 0) {
    appendHeading(panel, "Recent Logs", 4);
    const recentLogs = debugState.logs.slice(-10);
    recentLogs.forEach((log) => {
      const line = document.createElement("div");
      line.style.marginBottom = "3px";
      line.textContent = `${log.timestamp} [${log.category}]: ${log.message}`;
      panel.append(line);
    });
  }
}

function appendHeading(container, text, level) {
  const heading = document.createElement(`h${level}`);
  heading.textContent = text;
  container.append(heading);
}

function appendParagraph(container, text) {
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  container.append(paragraph);
}

function appendPreformatted(container, text) {
  const pre = document.createElement("pre");
  pre.textContent = text;
  container.append(pre);
}

function appendStatusLine(container, text, success) {
  const line = document.createElement("div");
  line.style.color = success ? "#66ff66" : "#ff6666";
  line.style.marginBottom = "5px";
  line.textContent = text;
  container.append(line);
}

// Create debug toggle button
function addDebugButton() {
  let btn = document.getElementById("sws-debug-btn");

  if (!btn) {
    btn = document.createElement("button");
    btn.id = "sws-debug-btn";
    btn.textContent = "🐞";
    btn.title = "Show Debug Panel";
    btn.style.cssText = `
      position: fixed;
      bottom: 10px;
      right: 10px;
      width: 30px;
      height: 30px;
      border-radius: 50%;
      background: rgba(0, 0, 0, 0.5);
      color: white;
      border: 1px solid #ccc;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9998;
    `;
    btn.onclick = () => {
      captureDomState();
      showDebugPanel();
    };

    document.body.appendChild(btn);
  }
}

function installFetchTracking() {
  if (window.fetch.swsDebugWrapped) {
    return;
  }

  const trackedFetch = function (...args) {
    const url = args[0] instanceof Request ? args[0].url : args[0];
    logDebug("FETCH", `Making fetch request to: ${url}`);

    return originalFetch
      .apply(this, args)
      .then((response) => {
        const success = response.ok;
        if (success) {
          logDebug("FETCH", `Successful fetch from: ${url}`);
          // Clone the response so we can both use it and read its body
          const clone = response.clone();
          clone
            .json()
            .then((data) => {
              trackApiCall(url, true, data, null);
            })
            .catch((_err) => {
              // Not JSON or other issue
              trackApiCall(url, true, { nonJsonResponse: true }, null);
            });
        } else {
          logError(
            `Fetch error: ${url} returned ${response.status}`,
            new Error(`HTTP ${response.status}`),
          );
          trackApiCall(url, false, null, new Error(`HTTP ${response.status}`));
        }
        return response;
      })
      .catch((error) => {
        logError(`Fetch failed: ${url}`, error);
        trackApiCall(url, false, null, error);
        throw error;
      });
  };

  trackedFetch.swsDebugWrapped = true;
  trackedFetch.swsOriginal = originalFetch;
  window.fetch = trackedFetch;
}

function installConsoleErrorTracking() {
  if (console.error.swsDebugWrapped) {
    return;
  }

  const trackedConsoleError = function (...args) {
    // Check if this is an error object
    const errorObjects = args.filter((arg) => arg instanceof Error);
    if (errorObjects.length > 0) {
      errorObjects.forEach((error) => {
        logError("Console error", error);
      });
    } else {
      // Regular console.error call
      logError("Console error", { message: args.join(" ") });
    }

    // Call the original
    originalConsoleError.apply(this, args);
  };

  trackedConsoleError.swsDebugWrapped = true;
  trackedConsoleError.swsOriginal = originalConsoleError;
  console.error = trackedConsoleError;
}

function installGlobalErrorTracking() {
  if (window.SWS_DEBUG_ERROR_TRACKING_INSTALLED) {
    return;
  }

  window.addEventListener("error", (event) => {
    logError("Uncaught error", {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      error: event.error,
    });
  });

  window.addEventListener("unhandledrejection", (event) => {
    logError("Unhandled Promise rejection", event.reason);
  });

  window.SWS_DEBUG_ERROR_TRACKING_INSTALLED = true;
}

// Initialization function
function initDebugger() {
  // Only initialize if debug is enabled
  if (!debugEnabled) {
    return;
  }

  logDebug("INIT", "Debug module initialized");
  addDebugButton();

  // Log initial details
  logDebug("CONFIG", "SWIMCONFIG state", window.SWIMCONFIG || "undefined");

  // Track DOM content loaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      logDebug("DOM", "DOMContentLoaded event fired");
      setTimeout(captureDomState, 100);
    });
  } else {
    logDebug("DOM", "DOM already loaded");
    setTimeout(captureDomState, 100);
  }

  // Track window load
  window.addEventListener("load", () => {
    logDebug("LOAD", "Window load event fired");
    setTimeout(captureDomState, 500);

    // Also capture after a delay to check if data loaded
    setTimeout(captureDomState, 3000);
  });
}

installFetchTracking();
installConsoleErrorTracking();
installGlobalErrorTracking();

// Initialize the debugger (will check internally if debug is enabled)
initDebugger();
