# Inconsistency Report: Shall We Swim Today?

This report highlights areas where the implementation deviates from the apparent design goals of the application, based on a review of the configuration, API, and application logic.

---

## **1. Partially Implemented Features for Multiple Locations**

- **Issue:** The application is designed to support multiple swimming locations, but some key features are hardcoded to work only for "nyc".
- **Evidence:**
  - In `shallweswim/api.py`, the `/api/{location}/currents` endpoint has a check that raises a `501 Not Implemented` error for any location other than "nyc". This indicates that the current prediction logic is specific to New York and has not been generalized.
  - The presence of numerous disabled locations in `shallweswim/config.py` (e.g., "chi", "sfo", "sdf") with complete or partial data source configurations suggests that the intent is to support these locations, but the implementation is not yet complete.

### **2. Incomplete User Experience Features**

- **Issue:** There are planned improvements to the user experience that have not been implemented.
- **Evidence:**
  - In `shallweswim/main.py`, the `root_index` function has a `TODO` comment: `TODO: Use cookies to redirect to last used or saved location.` This indicates a planned feature to personalize the user experience that is currently missing.

### **3. Stale or Experimental Code**

- **Issue:** There are commented-out code blocks and configurations that appear to be either deprecated or experimental, which can create confusion.
- **Evidence:**
  - In `shallweswim/config.py`, the configuration for "sfo" (San Francisco) has a commented-out `CoopsTempFeedConfig` with a note that the station is unavailable. While the note is helpful, the commented-out code itself could be removed to improve clarity.
  - The "tst" (Test) location in `shallweswim/config.py` seems to be for internal testing and is disabled, which is appropriate, but it adds to the number of incomplete location configurations in the file.

### **4. Potential for Configuration-Driven Logic**

- **Issue:** Some logic that is hardcoded in the API could be driven by the location configuration, which would make the system more flexible.
- **Evidence:**
  - The aforementioned check `if location != "nyc":` in `shallweswim/api.py` is a prime candidate for this. A more robust design might involve adding a `features` field to the `LocationConfig` model in `shallweswim/config.py` to indicate which features (e.g., "current_predictions") are available for each location. The API could then check for the presence of a feature flag instead of a hardcoded location code.

---

### **Process Note**

This report should be regenerated regularly (e.g., before a new release or at the beginning of a development cycle) to ensure that the codebase remains aligned with its design goals. As the project evolves, new inconsistencies may arise, and this report can serve as a valuable tool for identifying and prioritizing technical debt and refactoring opportunities.

### **How to Generate This Report**

To generate an updated version of this report, ensure the AI agent has access to the entire codebase and use a prompt similar to the following:

> "Please review the entire codebase for the 'Shall We Swim Today?' project. Your goal is to identify any inconsistencies between the implemented code and the apparent design goals or architecture. Look for things like partially implemented features, hardcoded logic that should be configurable, stale or experimental code, and any deviations from the patterns established in the core components. Create a report detailing your findings, providing specific evidence from the code to support each point."
