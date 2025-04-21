// Youtube Webcam Embed
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
function onYTPlayerReady(event) {
  console.log("onPlayerReady");
  // Autoplay will only work if video is already muted.
  event.target.mute();
  event.target.playVideo();
}
function onYTPlayerError(event) {
  console.log("onPlayerError");
}
function onYTPlayerStateChange(event) {
  console.log("onPlayerStateChange");
}

// Train status
const STATUSES = {
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

// Refresh interval in milliseconds
const REFRESH_INTERVAL = 60000; // 60 seconds
function load_trains() {
  get_train_status("Q");
  get_train_status("B");
}

// Initialize page with API data and set up refresh
function initializeWithApi() {
  // Get the location code from the page URL
  const pathParts = window.location.pathname.split("/");
  const location = pathParts[pathParts.length > 1 ? 1 : 0] || "nyc";

  // Load initial data
  fetchAndUpdateConditions(location);

  // Set up automatic refresh
  setInterval(() => {
    fetchAndUpdateConditions(location);
  }, REFRESH_INTERVAL);
}

// Fetch conditions data from API and update page
function fetchAndUpdateConditions(location) {
  fetch(`/api/${location}/conditions`)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      updatePageWithConditions(data);
    })
    .catch((error) => {
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
    });
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
    tempStationElement.textContent = `at ${data.location.name} as of ${formattedTime}.`;
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

// Format ISO date string to human-readable format
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
function display_value(element, value) {
  if (!value || (Array.isArray(value) && value.length == 0)) {
    document.getElementById(element + "_div").style.display = "none";
    return;
  }
  document.getElementById(element).textContent = value;
  document.getElementById(element + "_div").style.display = "block";
}
function get_train_status(train) {
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
      const status_class = STATUSES[status_str];

      document.getElementById(t + "_status").textContent = status_str;
      document.getElementById(t + "_status").className = status_class;

      if (data.status != "Not Scheduled") {
        document.getElementById(t + "_destination").textContent =
          data.destinations.south[0];
        display_value(t + "_delay", data.delay_summaries.south);
        const changes =
          data.service_change_summaries.both +
          data.service_change_summaries.south;
        display_value(t + "_service_change", changes);
        display_value(
          t + "_service_irregularity",
          data.service_irregularity_summaries.south,
        );
      }
    })
    .catch((error) => console.error(error));
}

// Initialize the page when it loads
function initPage() {
  // Load train information if needed
  load_trains();

  // Set up API-based updates
  initializeWithApi();

  // Optionally enable snow effect (currently commented out in HTML)
  /*make_snow();*/
}

// Initialize the embed page when it loads - no trains, just API data
function initEmbedPage() {
  // Default to NYC location for embed
  const location = "nyc";

  // Load initial data
  fetchAndUpdateConditions(location);

  // Set up automatic refresh
  setInterval(() => {
    fetchAndUpdateConditions(location);
  }, REFRESH_INTERVAL);
}

// Snow effect
function make_snow() {
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
