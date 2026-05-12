/**
 * ShallWeSwim - Main JavaScript
 *
 * Client-side data loading for condition, current, webcam, and transit displays.
 */

//=============================================================================
// CONFIGURATION
//=============================================================================

const REFRESH_INTERVAL = 60000; // 60 seconds
const DEFERRED_PLOT_RETRY_DELAYS = window.SWS_DEFERRED_PLOT_RETRY_DELAYS || [
  1000, 3000, 7000,
];

const TRANSIT_STATUS_COLORS = {
  Delay: "status-red",
  "No Service": "status-black",
  "Service Change": "status-orange",
  Slow: "status-yellow",
  "Not Good": "status-yellow",
  "Good Service": "status-green",
  "Not Scheduled": "status-white",
  "No Data": "status-white",
  Unavailable: "status-white",
};

let locationCode = getConfiguredLocationCode();
let conditionsRefreshTimer = null;
let conditionsFetchInFlight = false;
let conditionsLoaded = false;
let deferredPlotsStarted = false;
let currentsLoaded = false;
let pageInitialized = false;
let currentShiftParam = null;
const transitLoaded = {
  B: false,
  Q: false,
};

//=============================================================================
// YOUTUBE WEBCAM EMBED
//=============================================================================

window.onYouTubeIframeAPIReady = function onYouTubeIframeAPIReady() {
  console.log("Creating player");
  new YT.Player("bbcam_player", {
    playerVars: {
      autoplay: 1,
      playsinline: 1,
      fs: 1,
      controls: 0,
      iv_load_policy: 3,
      rel: 0,
    },
    events: {
      onReady: onYTPlayerReady,
      onError: onYTPlayerError,
      onStateChange: onYTPlayerStateChange,
    },
  });
  console.log("Created player");
};

function onYTPlayerReady(event) {
  console.log("onPlayerReady");
  event.target.mute();
  event.target.playVideo();
}

function onYTPlayerError(_event) {
  console.log("onPlayerError");
}

function onYTPlayerStateChange(_event) {
  console.log("onPlayerStateChange");
}

//=============================================================================
// PAGE INITIALIZATION
//=============================================================================

function getConfiguredLocationCode() {
  return window.SWIMCONFIG?.locationCode || null;
}

function refreshLocationCode() {
  locationCode = getConfiguredLocationCode();
  if (!locationCode) {
    console.warn("SWIMCONFIG location code is not available");
  }
  return locationCode;
}

function initializePage() {
  if (pageInitialized) {
    return;
  }
  pageInitialized = true;

  refreshLocationCode();

  if (window.location.pathname.includes("/currents")) {
    initCurrentsPage();
  } else if (window.location.pathname.includes("/embed")) {
    initEmbedPage();
  } else {
    initPage();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializePage);
} else {
  initializePage();
}

function startConditionsUpdates() {
  if (!locationCode) {
    return;
  }

  fetchAndUpdateConditions(locationCode);

  if (!conditionsRefreshTimer) {
    conditionsRefreshTimer = setInterval(() => {
      fetchAndUpdateConditions(locationCode);
    }, REFRESH_INTERVAL);
  }
}

function initPage() {
  startConditionsUpdates();

  if (window.location.pathname.includes("nyc")) {
    loadTransitStatus();
  }

  const month = new Date().getMonth();
  if (month === 11 || month === 0 || month === 1) {
    makeSnow();
  }
}

function initEmbedPage() {
  startConditionsUpdates();
}

function initCurrentsPage() {
  if (!locationCode) {
    console.error(
      "Cannot initialize currents page: No location code available",
    );
    return;
  }

  currentShiftParam = getShiftParam();
  fetchCurrentsData(locationCode, currentShiftParam);
}

//=============================================================================
// CONDITIONS API DATA HANDLING
//=============================================================================

async function fetchAndUpdateConditions(location) {
  if (!location) {
    console.error("Cannot fetch conditions: No location specified");
    return;
  }

  if (conditionsFetchInFlight) {
    return;
  }

  conditionsFetchInFlight = true;
  showConditionsLoading();

  try {
    console.log(`Fetching conditions data for ${location}...`);
    const response = await fetch(`/api/${location}/conditions`);

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    console.log(`Successfully loaded conditions data for ${location}`);
    updatePageWithConditions(data);
  } catch (error) {
    console.error("Error fetching conditions:", error);
    showConditionsError();
  } finally {
    conditionsFetchInFlight = false;
    startDeferredPlotLoading();
  }
}

function startDeferredPlotLoading() {
  if (deferredPlotsStarted) {
    return;
  }
  deferredPlotsStarted = true;
  loadDeferredPlots();
}

function showConditionsLoading() {
  if (!conditionsLoaded) {
    clearConditionsStatus();
  }
}

function showConditionsError() {
  if (conditionsLoaded) {
    setConditionsStatus(
      "Could not refresh latest conditions. Showing last loaded data.",
    );
    return;
  }

  updateTemperatureUnavailable();
  updateTidesUnavailable();
  updateCurrentUnavailable();
  setConditionsStatus(
    "Unable to load latest conditions. Please try again later.",
  );
}

function updatePageWithConditions(data) {
  updateTemperature(data);
  updateTides(data.tides);
  updateCurrent(data.current);
  conditionsLoaded = true;
  clearConditionsStatus();

  console.log(
    "Page updated with fresh data from API at",
    new Date().toLocaleTimeString(),
  );
}

function updateTemperature(data) {
  const temperature = data.temperature;
  const tempElement = document.getElementById("water-temp");
  const tempStationElement = document.getElementById("temp-station-info");

  if (!tempElement && !tempStationElement) {
    return;
  }

  if (!temperature || temperature.water_temp === null) {
    updateTemperatureUnavailable();
    return;
  }

  if (tempElement) {
    tempElement.textContent = `${temperature.water_temp}°${temperature.units || "F"}`;
  }

  if (tempStationElement) {
    const stationName =
      temperature.station_name || data.location?.name || "station";
    renderTemperatureStationInfo(
      tempStationElement,
      stationName,
      temperature.timestamp,
    );
  }
}

function updateTemperatureUnavailable() {
  setText("water-temp", "Unavailable");

  const tempStationElement = document.getElementById("temp-station-info");
  if (tempStationElement) {
    tempStationElement.textContent =
      "Current water temperature is unavailable.";
  }
}

function renderTemperatureStationInfo(container, stationName, timestamp) {
  container.textContent = "";
  container.append("at ");

  const stationSpan = document.createElement("span");
  stationSpan.id = "temp-station-name";
  stationSpan.textContent = stationName;
  container.append(stationSpan);

  const formattedTime = formatTimestamp(timestamp, {
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  });

  if (formattedTime) {
    container.append(` as of ${formattedTime}.`);
  } else {
    container.append(".");
  }
}

function updateTides(tides) {
  const tidesSection = document.getElementById("tides-section");
  if (!tidesSection) {
    return;
  }

  if (!tides) {
    updateTidesUnavailable();
    return;
  }

  const pastTide = tides.past?.at(-1);
  updateTideRow("past-tide", pastTide);

  updateTideRow("next-tide-0", tides.next?.[0]);
  updateTideRow("next-tide-1", tides.next?.[1]);
}

function updateTideRow(prefix, tide) {
  setText(`${prefix}-type`, tide?.type || "Unavailable");
  setText(`${prefix}-date`, tide ? formatDate(tide.time) : "Unavailable");
  setText(`${prefix}-time`, tide ? formatTime(tide.time) : "Unavailable");
}

function updateTidesUnavailable() {
  updateTideRow("past-tide", null);
  updateTideRow("next-tide-0", null);
  updateTideRow("next-tide-1", null);
}

function updateCurrent(current) {
  const currentMagnitudeElement = document.getElementById("current-magnitude");
  const currentStateSummaryElement = document.getElementById(
    "current-state-summary",
  );
  const currentDetailsLink = document.getElementById("current-details-link");

  if (!current) {
    updateCurrentUnavailable();
    return;
  }

  const magnitude = Number.parseFloat(current.magnitude);
  setElementText(
    currentMagnitudeElement,
    Number.isFinite(magnitude) ? magnitude.toFixed(1) : "N/A",
  );
  setElementText(
    currentStateSummaryElement,
    current.state_description || current.direction?.toLowerCase() || "flowing",
  );

  if (currentDetailsLink) {
    currentDetailsLink.style.display = current.direction ? "" : "none";
  }
}

function updateCurrentUnavailable() {
  setText("current-magnitude", "N/A");
  setText("current-state-summary", "unavailable");

  const currentDetailsLink = document.getElementById("current-details-link");
  if (currentDetailsLink) {
    currentDetailsLink.style.display = "none";
  }
}

function setConditionsStatus(message) {
  const statusElement = document.getElementById("conditions-status");
  if (statusElement) {
    statusElement.textContent = message;
    statusElement.hidden = false;
  }
}

function clearConditionsStatus() {
  const statusElement = document.getElementById("conditions-status");
  if (statusElement) {
    statusElement.textContent = "";
    statusElement.hidden = true;
  }
}

//=============================================================================
// DEFERRED PLOT LOADING
//=============================================================================

function loadDeferredPlots() {
  const plots = document.querySelectorAll("img.deferred-plot[data-src]");
  plots.forEach((plot) => {
    loadDeferredPlot(plot);
  });
}

function loadDeferredPlot(plot, attempt = 0) {
  const src = plot.dataset.src;
  if (!src || plot.dataset.loaded === "true") {
    return;
  }

  plot.dataset.status = "loading";

  const probe = new Image();
  probe.onload = () => {
    plot.src = src;
    plot.dataset.loaded = "true";
    plot.dataset.status = "loaded";
    hideDeferredPlotStatus(plot);
  };
  probe.onerror = () => {
    const retryDelay = DEFERRED_PLOT_RETRY_DELAYS[attempt];
    if (retryDelay === undefined) {
      plot.dataset.status = "unavailable";
      showDeferredPlotStatus(plot, "Plot unavailable");
      return;
    }

    plot.dataset.status = "retrying";
    window.setTimeout(() => loadDeferredPlot(plot, attempt + 1), retryDelay);
  };
  probe.src = src;
}

function showDeferredPlotStatus(plot, message) {
  const statusElement = getDeferredPlotStatusElement(plot);
  statusElement.textContent = message;
  statusElement.hidden = false;
}

function hideDeferredPlotStatus(plot) {
  const statusElement = getDeferredPlotStatusElement(plot);
  statusElement.textContent = "";
  statusElement.hidden = true;
}

function getDeferredPlotStatusElement(plot) {
  const nextElement = plot.nextElementSibling;
  if (nextElement?.classList.contains("plot-status")) {
    return nextElement;
  }

  const statusElement = document.createElement("div");
  statusElement.className = "plot-status note";
  statusElement.hidden = true;
  plot.insertAdjacentElement("afterend", statusElement);
  return statusElement;
}

function setText(id, value) {
  setElementText(document.getElementById(id), value);
}

function setElementText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function formatTimestamp(isoString, options) {
  if (!isoString) {
    return "";
  }

  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleString("en-US", options);
}

function formatDate(isoString) {
  return formatTimestamp(isoString, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

function formatTime(isoString) {
  return formatTimestamp(isoString, {
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  });
}

//=============================================================================
// TRANSIT STATUS
//=============================================================================

function loadTransitStatus() {
  getTrainStatus("Q");
  getTrainStatus("B");
}

function displayValue(element, value) {
  const valueElement = document.getElementById(element);
  const container = document.getElementById(`${element}_div`);
  if (!valueElement || !container) {
    return;
  }

  if (!value || (Array.isArray(value) && value.length === 0)) {
    container.style.display = "none";
    return;
  }

  valueElement.textContent = value;
  container.style.display = "block";
}

function clearTrainAlerts(train) {
  displayValue(`${train}_delay`, null);
  displayValue(`${train}_service_change`, null);
  displayValue(`${train}_service_irregularity`, null);
}

function setTrainStatus(train, status) {
  const statusElement = document.getElementById(`${train}_status`);
  if (!statusElement) {
    return;
  }

  statusElement.textContent = status;
  statusElement.className = TRANSIT_STATUS_COLORS[status] || "status-white";
}

function updateTrainUnavailable(train) {
  if (transitLoaded[train]) {
    console.warn(
      `Could not refresh ${train} train status; keeping prior data.`,
    );
    return;
  }

  setTrainStatus(train, "Unavailable");
  setText(`${train}_destination`, "unavailable");
  clearTrainAlerts(train);
}

function getTrainStatus(train) {
  const t = train;
  fetch(`https://goodservice.io/api/routes/${train}`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(
          `Transit API request failed with status ${response.status}`,
        );
      }
      return response.json();
    })
    .then((data) => {
      let statusStr = "No Data";
      if (data.status === "Not Scheduled") {
        statusStr = "Not Scheduled";
      } else if (data.direction_statuses?.south) {
        statusStr = data.direction_statuses.south;
      }

      setTrainStatus(t, statusStr);
      clearTrainAlerts(t);

      if (data.status !== "Not Scheduled") {
        setText(`${t}_destination`, data.destinations?.south?.[0] || "unknown");
        displayValue(`${t}_delay`, data.delay_summaries?.south);

        const changes = [
          data.service_change_summaries?.both,
          data.service_change_summaries?.south,
        ]
          .filter(Boolean)
          .join("");
        displayValue(`${t}_service_change`, changes);
        displayValue(
          `${t}_service_irregularity`,
          data.service_irregularity_summaries?.south,
        );
      }
      transitLoaded[t] = true;
    })
    .catch((error) => {
      console.warn(error);
      updateTrainUnavailable(t);
    });
}

//=============================================================================
// CURRENTS PAGE FUNCTIONALITY
//=============================================================================

async function fetchCurrentsData(location, shift = null) {
  if (!location) {
    console.error("Cannot fetch currents: No location specified");
    return;
  }

  showCurrentsLoading();

  try {
    console.log(
      `Fetching currents data for ${location}${shift ? ` with shift ${shift}` : ""}...`,
    );

    const url = new URL(`/api/${location}/currents`, window.location.origin);
    if (shift) {
      url.searchParams.set("shift", shift);
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    updateCurrentsDisplay(data, location);
  } catch (error) {
    console.error("Error fetching currents data:", error);
    showCurrentsError();
  }
}

function updateCurrentsDisplay(data, location) {
  currentsLoaded = true;
  clearCurrentsStatus();

  setText("timestamp", formatTimestamp(data.timestamp) || "unavailable");

  const magnitude = Number.parseFloat(data.current?.magnitude);
  setText(
    "magnitude",
    Number.isFinite(magnitude) ? magnitude.toFixed(1) : "N/A",
  );
  setText(
    "state",
    data.current?.state_description ||
      data.current?.direction?.toLowerCase() ||
      "flowing",
  );

  const prevLink = document.getElementById("prev-hour-link");
  const nextLink = document.getElementById("next-hour-link");

  if (prevLink && data.navigation?.prev_hour !== undefined) {
    prevLink.href = `/${location}/currents?shift=${data.navigation.prev_hour}`;
  }

  if (nextLink && data.navigation?.next_hour !== undefined) {
    nextLink.href = `/${location}/currents?shift=${data.navigation.next_hour}`;
  }

  updateCharts(data);
}

function showCurrentsLoading() {
  if (!currentsLoaded) {
    clearCurrentsStatus();
  }
}

function showCurrentsError() {
  if (currentsLoaded) {
    setCurrentsStatus(
      "Could not refresh current prediction. Showing last loaded data.",
    );
    return;
  }

  setText("timestamp", "unavailable");
  setText("state", "unavailable");
  setText("magnitude", "N/A");
  setCurrentsStatus(
    "Unable to load current prediction. Please try again later.",
  );
}

function setCurrentsStatus(message) {
  const statusElement = document.getElementById("currents-status");
  if (statusElement) {
    statusElement.textContent = message;
    statusElement.hidden = false;
  }
}

function clearCurrentsStatus() {
  const statusElement = document.getElementById("currents-status");
  if (statusElement) {
    statusElement.textContent = "";
    statusElement.hidden = true;
  }
}

function updateCharts(data) {
  const currentMapSection = document.getElementById("current-map-section");
  const currentChart = document.getElementById("current-chart");
  if (data.current_chart_filename) {
    if (currentMapSection) {
      currentMapSection.hidden = false;
    }
    if (currentChart) {
      currentChart.src = data.current_chart_filename;
    }
  } else {
    if (currentMapSection) {
      currentMapSection.hidden = true;
    }
    if (currentChart) {
      currentChart.removeAttribute("src");
    }
  }

  const tideCurrentPlot = document.getElementById("tide-current-plot");
  if (tideCurrentPlot) {
    const shift = data.navigation?.shift || 0;
    tideCurrentPlot.src = `/api/${data.location.code}/plots/current_tide?shift=${shift}`;
  }

  const legacyMapSection = document.getElementById("legacy-map-section");
  const legacyChart = document.getElementById("legacy-chart");
  const legacyMapTitle = document.getElementById("legacy-map-title");
  if (data.legacy_chart?.chart_filename) {
    if (legacyMapSection) {
      legacyMapSection.hidden = false;
    }
    if (legacyChart) {
      legacyChart.src = `/static/tidecharts/${data.legacy_chart.chart_filename}`;
    }
    if (legacyMapTitle) {
      legacyMapTitle.textContent = `Legacy Map - ${data.legacy_chart.map_title}`;
    }
  } else {
    if (legacyMapSection) {
      legacyMapSection.hidden = true;
    }
    if (legacyChart) {
      legacyChart.removeAttribute("src");
    }
    if (legacyMapTitle) {
      legacyMapTitle.textContent = "Legacy Map";
    }
  }
}

function getShiftParam() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("shift");
}

//=============================================================================
// SPECIAL EFFECTS
//=============================================================================

function makeSnow() {
  if (document.getElementById("particles-js-script")) {
    return;
  }

  const script = document.createElement("script");
  script.id = "particles-js-script";
  script.src = "https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js";
  script.onload = () => {
    particlesJS("snow", {
      particles: {
        number: {
          value: 200,
          density: {
            enable: true,
            value_area: 800,
          },
        },
        color: {
          value: "#ffffff",
        },
        opacity: {
          value: 0.7,
          random: false,
          anim: {
            enable: false,
          },
        },
        size: {
          value: 5,
          random: true,
          anim: {
            enable: false,
          },
        },
        line_linked: {
          enable: false,
        },
        move: {
          enable: true,
          speed: 1,
          direction: "bottom",
          random: true,
          straight: false,
          out_mode: "out",
          bounce: false,
          attract: {
            enable: true,
            rotateX: 300,
            rotateY: 1200,
          },
        },
      },
      interactivity: {
        events: {
          onhover: {
            enable: false,
          },
          onclick: {
            enable: false,
          },
          resize: false,
        },
      },
      retina_detect: true,
    });
  };
  document.head.append(script);
}
