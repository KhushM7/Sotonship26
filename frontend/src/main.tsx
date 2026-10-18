import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { Log } from "./views/Log";
import { Phone } from "./views/Phone";
import { Presenter } from "./views/Presenter";
import { Wall } from "./views/Wall";

// four pages, picked by path; the server serves index.html for all of them
function App() {
  const path = location.pathname;
  if (path.startsWith("/phone")) return <Phone path={path} />;
  if (path.startsWith("/presenter")) return <Presenter />;
  if (path.startsWith("/log")) return <Log />;
  return <Wall />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
