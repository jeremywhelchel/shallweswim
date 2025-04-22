/**
 * ShallWeSwim - Main JavaScript
 *
 * This file contains all client-side functionality for the ShallWeSwim app,
 * including API data handling, webcam embed, and transit information.
 */

//=============================================================================
// CONFIGURATION
//=============================================================================

// Refresh interval in milliseconds
const REFRESH_INTERVAL = 60000; // 60 seconds

// Transit status color mapping
const TRANSIT_STATUS_COLORS = {
  Delay: "status-red",
  "No Service": "status-black",
  "Service Change": "status-orange",
  Slow: "status-yellow",
  "Not Good": "status-yellow",
  "Good Service": "status-green",
  "Not Scheduled": "status-white",
  "No Data": "status-white",
  // 'Not Scheduled': 'status-black',
  // 'No Data': 'status-black',
};

//=============================================================================
// YOUTUBE WEBCAM EMBED
//=============================================================================

/**
 * Initializes the YouTube player when the API is ready
 */
function onYouTubeIframeAPIReady() {
  console.log("Creating player");
  var bbcam_player;
  bbcam_player = new YT.Player("bbcam_player", {
    playerVars: {
      autoplay: 1,
      playsinline: 1,
      fs: 1,
      controls: 0,
      iv_load_policy: 3, // disable video annotations
      rel: 0,
    },
    events: {
      onReady: onYTPlayerReady,
      onError: onYTPlayerError,
      onStateChange: onYTPlayerStateChange,
    },
  });
  console.log("Created player");
}

/**
 * Handles YouTube player ready event
 */
function onYTPlayerReady(event) {
  console.log("onPlayerReady");
  // Autoplay will only work if video is already muted.
  event.target.mute();
  event.target.playVideo();
}

/**
 * Handles YouTube player errors
 */
function onYTPlayerError(event) {
  console.log("onPlayerError");
}

/**
 * Handles YouTube player state changes
 */
function onYTPlayerStateChange(event) {
  console.log("onPlayerStateChange");
}

//=============================================================================
// TRANSIT STATUS
//=============================================================================

/**
 * Loads status information for NYC transit lines
 */
function loadTransitStatus() {
  getTrainStatus("Q");
  getTrainStatus("B");
}

//=============================================================================
// API DATA HANDLING
//=============================================================================

// Global variable for storing the location code
let locationCode = "nyc"; // Default fallback

// Get the location code as early as possible, even before DOM is ready
if (window.SWIMCONFIG && window.SWIMCONFIG.locationCode) {
  locationCode = window.SWIMCONFIG.locationCode;
  console.log(`Location code set to ${locationCode} from SWIMCONFIG (early)`);
}

// Try to fetch data immediately if SWIMCONFIG is available early
if (window.SWIMCONFIG && window.SWIMCONFIG.locationCode) {
  console.log("Starting early data fetch...");
  // We're doing this before DOMContentLoaded for faster initial load
  fetchAndUpdateConditions(locationCode);
}

// This function will be called on both DOMContentLoaded and window.load
function initializeLocationAndData() {
  console.log("Initializing location and data...");
  // Get/confirm the location code from the global SWIMCONFIG variable
  if (window.SWIMCONFIG && window.SWIMCONFIG.locationCode) {
    locationCode = window.SWIMCONFIG.locationCode;
    console.log(`Location code set to ${locationCode} from SWIMCONFIG`);
  } else {
    console.warn("SWIMCONFIG not available, using default location");
  }

  // Make sure we have fresh data (will be ignored if an identical request is in flight)
  fetchAndUpdateConditions(locationCode);
  return locationCode;
}

// Initialize location code and fetch data when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", initializeLocationAndData);

// Also initialize when the window loads (helps with refresh issues)
window.addEventListener("load", initializeLocationAndData);

/**
 * Initialize page with API data and set up refresh
 */
function initializeWithApi() {
  // Set up automatic refresh only (initial fetch already started)
  setInterval(() => {
    fetchAndUpdateConditions(locationCode);
  }, REFRESH_INTERVAL);
}

// Fetch conditions data from API and update page
async function fetchAndUpdateConditions(location) {
  try {
    console.log(`Fetching conditions data for ${location}...`);
    // Using the same pattern as the currents page which works on Safari
    const response = await fetch(`/api/${location}/conditions`);

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    console.log(`Successfully loaded conditions data for ${location}`);
    updatePageWithConditions(data);
  } catch (error) {
    console.error("Error fetching conditions:", error);

    // Update UI even on error to avoid infinite "Loading..." state
    const tempElement = document.getElementById("water-temp");
    if (tempElement && tempElement.textContent === "Loading...") {
      tempElement.textContent = "Data unavailable";
    }

    const tempStationElement = document.getElementById("temp-station-info");
    if (tempStationElement) {
      tempStationElement.textContent =
        "Unable to retrieve current data. Please try again later.";
    }
  }
}

// Update the page with data from the API
function updatePageWithConditions(data) {
  // Update water temperature
  const tempElement = document.getElementById("water-temp");
  if (tempElement) {
    tempElement.textContent = `${data.temperature.water_temp}°F`;
  }

  // Update temperature station info
  const tempStationElement = document.getElementById("temp-station-info");
  if (tempStationElement) {
    const timestamp = new Date(data.temperature.timestamp);
    const options = {
      month: "long",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    };
    const formattedTime = timestamp.toLocaleDateString("en-US", options);
    // Use the temperature station name from the API if available, otherwise fallback to location name
    const stationName = data.temperature.station_name || data.location.name;
    tempStationElement.textContent = `at ${stationName} as of ${formattedTime}.`;
  }

  // Update past tide information
  if (data.tides.past.length > 0) {
    const pastTide = data.tides.past[data.tides.past.length - 1];

    // Update past tide type
    const pastTideTypeElement = document.getElementById("past-tide-type");
    if (pastTideTypeElement) {
      pastTideTypeElement.textContent = pastTide.type;
    }

    // Update past tide time
    const pastTideTimeElement = document.getElementById("past-tide-time");
    if (pastTideTimeElement) {
      pastTideTimeElement.textContent = formatDateTime(pastTide.time);
    }
  }

  // Update next tides information
  if (data.tides.next.length > 0) {
    // Update first next tide
    if (data.tides.next.length >= 1) {
      const nextTide0TypeElement = document.getElementById("next-tide-0-type");
      if (nextTide0TypeElement) {
        nextTide0TypeElement.textContent = data.tides.next[0].type;
      }

      const nextTide0TimeElement = document.getElementById("next-tide-0-time");
      if (nextTide0TimeElement) {
        nextTide0TimeElement.textContent = formatDateTime(
          data.tides.next[0].time,
        );
      }
    }

    // Update second next tide
    if (data.tides.next.length >= 2) {
      const nextTide1TypeElement = document.getElementById("next-tide-1-type");
      if (nextTide1TypeElement) {
        nextTide1TypeElement.textContent = data.tides.next[1].type;
      }

      const nextTide1TimeElement = document.getElementById("next-tide-1-time");
      if (nextTide1TimeElement) {
        nextTide1TimeElement.textContent = formatDateTime(
          data.tides.next[1].time,
        );
      }
    }
  }

  // Log the update time in console for debugging
  console.log(
    "Page updated with fresh data from API at",
    new Date().toLocaleTimeString(),
  );
}

/**
 * Format ISO date string to human-readable format
 * @param {string} isoString - ISO8601 datetime string
 * @returns {string} Formatted date string
 */
function formatDateTime(isoString) {
  const date = new Date(isoString);
  const options = {
    weekday: "long",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "numeric",
    hour12: true,
  };
  return date.toLocaleDateString("en-US", options);
}
/**
 * Utility to display a value in the DOM or hide its container if empty
 * @param {string} element - The base ID of the element
 * @param {*} value - The value to display
 */
function displayValue(element, value) {
  if (!value || (Array.isArray(value) && value.length == 0)) {
    document.getElementById(element + "_div").style.display = "none";
    return;
  }
  document.getElementById(element).textContent = value;
  document.getElementById(element + "_div").style.display = "block";
}
/**
 * Fetches and displays the status for a specific train line
 * @param {string} train - The train line code (e.g., "Q", "B")
 */
function getTrainStatus(train) {
  const t = train; //.toLowerCase();
  fetch("https://goodservice.io/api/routes/" + train)
    .then((response) => response.json())
    .then((data) => {
      status_str = "...";
      if (data.status == "Not Scheduled") {
        status_str = "Not Scheduled";
      } else {
        status_str = data.direction_statuses.south;
      }
      const status_class = TRANSIT_STATUS_COLORS[status_str];

      document.getElementById(t + "_status").textContent = status_str;
      document.getElementById(t + "_status").className = status_class;

      if (data.status != "Not Scheduled") {
        document.getElementById(t + "_destination").textContent =
          data.destinations.south[0];
        displayValue(t + "_delay", data.delay_summaries.south);
        const changes =
          data.service_change_summaries.both +
          data.service_change_summaries.south;
        displayValue(t + "_service_change", changes);
        displayValue(
          t + "_service_irregularity",
          data.service_irregularity_summaries.south,
        );
      }
    })
    .catch((error) => console.error(error));
}

/**
 * Initialize the main page when it loads
 */
function initPage() {
  // Set up automatic refresh (initial fetch already started)
  initializeWithApi();

  // Load transit status for NYC location
  if (window.location.pathname.includes("nyc")) {
    loadTransitStatus();
  }

  // Add snow effect during winter months (Dec-Feb)
  const now = new Date();
  const month = now.getMonth();
  if (month === 11 || month === 0 || month === 1) {
    makeSnow();
  }
}

/**
 * Initialize the embed page - no trains, just API data
 */
function initEmbedPage() {
  // Only set up refresh interval (data fetch already started)
  initializeWithApi();
}

//=============================================================================
// CURRENTS PAGE FUNCTIONALITY
//=============================================================================

/**
 * Fetches current prediction data from the API
 * @param {string} locationCode - Location code (e.g., "nyc")
 * @param {string|null} shift - Optional time shift parameter
 */
async function fetchCurrentsData(locationCode, shift = null) {
  try {
    // Build the API URL with optional shift parameter
    let url = `/api/${locationCode}/currents`;
    if (shift) {
      url += `?shift=${shift}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    updateCurrentsDisplay(data, locationCode);
  } catch (error) {
    console.error("Error fetching currents data:", error);
  }
}

/**
 * Updates the display with currents data
 * @param {Object} data - The currents data from the API
 * @param {string} locationCode - Location code for building navigation URLs
 */
function updateCurrentsDisplay(data, locationCode) {
  // Update timestamp
  const timestampElement = document.getElementById("timestamp");
  if (timestampElement) {
    timestampElement.textContent = new Date(data.timestamp).toLocaleString();
  }

  // Update current direction, magnitude and state
  const directionElement = document.getElementById("direction");
  if (directionElement) {
    directionElement.textContent = data.current.direction;
  }

  const magnitudeElement = document.getElementById("magnitude");
  if (magnitudeElement) {
    magnitudeElement.textContent = data.current.magnitude.toFixed(1);
  }

  const stateElement = document.getElementById("state");
  if (stateElement) {
    stateElement.textContent = data.current.state_description;
  }

  // Update navigation links
  const prevLink = document.getElementById("prev-hour-link");
  const nextLink = document.getElementById("next-hour-link");

  if (prevLink) {
    prevLink.href = `/${locationCode}/currents?shift=${data.navigation.prev_hour}`;
  }

  if (nextLink) {
    nextLink.href = `/${locationCode}/currents?shift=${data.navigation.next_hour}`;
  }

  // Update charts if that function exists
  if (typeof updateCharts === "function") {
    updateCharts(data);
  }
}

/**
 * Updates chart images when current data is loaded
 * @param {Object} data - The currents data from the API
 */
function updateCharts(data) {
  // Update current chart
  const currentChart = document.getElementById("current-chart");
  if (currentChart) {
    currentChart.src = data.current_chart_filename;
  }

  // Update tide current plot
  const tideCurrentPlot = document.getElementById("tide-current-plot");
  if (tideCurrentPlot) {
    tideCurrentPlot.src = `/api/${data.location.code}/current_tide_plot?shift=${data.navigation.shift}`;
  }

  // Update legacy chart
  const legacyChart = document.getElementById("legacy-chart");
  if (legacyChart) {
    legacyChart.src = `/static/tidecharts/${data.legacy_chart.chart_filename}`;
  }

  // Update legacy map title
  const legacyMapTitle = document.getElementById("legacy-map-title");
  if (legacyMapTitle) {
    legacyMapTitle.textContent = `Legacy Map - ${data.legacy_chart.map_title}`;
  }
}

/**
 * Initialize the currents page
 */
function initCurrentsPage() {
  // Use the global locationCode variable

  // Get shift parameter from URL if present
  const urlParams = new URLSearchParams(window.location.search);
  const shift = urlParams.get("shift");

  // Immediately fetch data (don't wait for DOMContentLoaded)
  fetchCurrentsData(locationCode, shift);
}

//=============================================================================
// AUTO-INITIALIZATION
//=============================================================================
/**
 * Automatically determine and initialize the correct page type when the DOM is loaded
 */
document.addEventListener("DOMContentLoaded", function () {
  // Check if we're on the currents page
  if (window.location.pathname.includes("/currents")) {
    console.log("Initializing currents page");
    initCurrentsPage();
  }
  // Check if we're on the embed page
  else if (window.location.pathname.includes("/embed")) {
    console.log("Initializing embed page");
    initEmbedPage();
  }
  // Also handle the index page here
  else {
    console.log("Initializing main page");
    initPage();
  }
});

/**
 * Creates a snow effect using particles.js
 */
function makeSnow() {
  var script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js";
  script.onload = function () {
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
