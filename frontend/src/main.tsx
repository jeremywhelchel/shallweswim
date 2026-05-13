import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./routes/App";
import "./styles/app.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element #root was not found");
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
