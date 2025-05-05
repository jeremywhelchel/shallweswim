/**
 * ShallWeSwim Debug Utility
 *
 * This script helps diagnose JavaScript and API loading issues across browsers.
 * It logs important events and can be activated by users with issues.
 */

// Debug state
const debugState = {
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

// Store original console functions
const originalConsoleError = console.error;
const originalConsoleLog = console.log;

// Add a log entry with timestamp
function logDebug(category, message, data = null) {
  const entry = {
    timestamp: new Date().toISOString(),
    category,
    message,
    data: data ? JSON.stringify(data).substring(0, 500) : null, // Truncate large data
  };
  debugState.logs.push(entry);
  // Use original console.log to avoid recursion if console.log is also patched
  originalConsoleLog.apply(console, [
    `[DEBUG:${category}] ${message}`,
    data || "",
  ]);

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
  // Use original console.error to prevent recursion
  originalConsoleError.apply(console, [`[ERROR] ${message}`, error]);

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

  let html = `<h3>Debug Info (${new Date().toLocaleTimeString()})</h3>`;

  // Browser info
  html += `<h4>Browser Information</h4>`;
  html += `<p>User Agent: ${debugState.browserInfo.userAgent}</p>`;
  html += `<p>Window Size: ${debugState.browserInfo.windowDimensions}</p>`;
  html += `<p>Online: ${debugState.browserInfo.onLine}</p>`;

  // DOM state
  html += `<h4>DOM State</h4>`;
  html += `<pre>${JSON.stringify(debugState.domState, null, 2)}</pre>`;

  // Most recent errors (max 3)
  if (debugState.errors.length > 0) {
    html += `<h4>Recent Errors</h4>`;
    const recentErrors = debugState.errors.slice(-3);
    recentErrors.forEach((err) => {
      html += `<div style="color: #ff6666; margin-bottom: 5px;">`;
      html += `${err.timestamp}: ${err.message} - ${err.errorName}: ${err.errorMessage}`;
      html += `</div>`;
    });
  }

  // Most recent API calls (max 3)
  if (debugState.apiCalls.length > 0) {
    html += `<h4>Recent API Calls</h4>`;
    const recentCalls = debugState.apiCalls.slice(-3);
    recentCalls.forEach((call) => {
      html += `<div style="color: ${call.success ? "#66ff66" : "#ff6666"}; margin-bottom: 5px;">`;
      html += `${call.timestamp}: ${call.url} - ${call.success ? "SUCCESS" : "FAILED"}`;
      html += `</div>`;
    });
  }

  // Most recent logs (max 10)
  if (debugState.logs.length > 0) {
    html += `<h4>Recent Logs</h4>`;
    const recentLogs = debugState.logs.slice(-10);
    recentLogs.forEach((log) => {
      html += `<div style="margin-bottom: 3px;">`;
      html += `${log.timestamp} [${log.category}]: ${log.message}`;
      html += `</div>`;
    });
  }

  panel.innerHTML = html;
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

// Monkey patch the fetch API to track calls
const originalFetch = window.fetch;
window.fetch = function (...args) {
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
          .catch((err) => {
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

// Monkey patch console.error to catch JS errors
console.error = function (...args) {
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

// Track global errors
window.addEventListener("error", function (event) {
  logError("Uncaught error", {
    message: event.message,
    filename: event.filename,
    lineno: event.lineno,
    colno: event.colno,
    error: event.error,
  });
});

// Track unhandled promise rejections
window.addEventListener("unhandledrejection", function (event) {
  logError("Unhandled Promise rejection", event.reason);
});

// Check if debugging is enabled via URL parameter
function isDebugEnabled() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("debug") === "1";
}

// Initialization function
function initDebugger() {
  // Only initialize if debug is enabled
  if (!isDebugEnabled()) {
    // Just set up error tracking without UI elements
    console.log("Debug mode disabled. Add ?debug=1 to URL to enable debugger.");
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

// Initialize the debugger (will check internally if debug is enabled)
initDebugger();
