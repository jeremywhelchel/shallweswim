import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        swim: {
          blue: "#000099",
          ink: "#172033",
          mist: "#fcffff",
          line: "#d6e4e8",
          current: "#006b8f",
          tide: "#5b7f2a",
          alert: "#a64b00",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
      },
    },
  },
  plugins: [],
} satisfies Config;
