import type { Incident, Level, Posture } from "./contracts";

export const POSTURE_LABEL: Record<Posture, string> = {
  upright: "Upright",
  falling: "Falling",
  on_floor: "On the floor",
  low_on_purpose: "Sitting or lying",
  not_visible: "Out of view",
};

export function levelText(level: Level, reason = "pose") {
  switch (level) {
    case "FALL":
      return "Fall detected";
    case "ON FLOOR":
      return "On the floor, has not got up";
    case "CHECK":
      return reason === "hidden_down"
        ? "Went out of view low down"
        : "Possible fall, please check";
    case "OFFLINE":
      return "Camera offline";
  }
}

export function incidentReason(i: Incident) {
  return i.levels.at(-1)?.reason ?? "pose";
}

/** 0:07, 4:32, 1:02:10 */
export function clock(seconds: number) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

/** "12 s", "4 min", "2 h 5 min" */
export function ago(seconds: number) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function timeOfDay(t: number) {
  return new Date(t * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export const STATE_LABEL: Record<string, string> = {
  open: "Waiting for a carer",
  escalated: "No response, escalated",
  acknowledged: "On the way",
  attending: "With the resident",
  resolved: "Resolved",
};

export const OUTCOME_LABEL: Record<string, string> = {
  real_fall: "Real fall",
  near_miss: "Near miss",
  false_alarm: "False alarm",
  camera_restored: "Camera restored",
};

export function firstName(name: string | null) {
  return name ? name.split(" ")[0] : "";
}

/** Is this incident still waiting for someone to respond? */
export const unanswered = (i: Incident) => i.state === "open" || i.state === "escalated";
export const urgent = (i: Incident) =>
  unanswered(i) && (i.level === "FALL" || i.level === "ON FLOOR");
