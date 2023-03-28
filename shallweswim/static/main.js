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
