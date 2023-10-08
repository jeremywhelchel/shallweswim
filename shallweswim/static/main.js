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
function load_trains(trains) {
  get_train_status("Q");
  get_train_status("B");
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
