import { defineConfig } from "vite";

// relative base: the same build works on localhost, GitHub Pages or any sub-folder
export default defineConfig({
  base: "./",
  server: { port: 4200, strictPort: true },
  preview: { port: 4200 },
});
