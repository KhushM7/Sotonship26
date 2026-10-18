import { create } from "zustand";
import { identity } from "./api";
import type {
  FrameMsg,
  Hello,
  Incident,
  LogEvent,
  Posture,
  RoomStatus,
  ScoresMsg,
  ServerMsg,
} from "./contracts";

/**
 * Fast-changing data (skeletons at 15 Hz, score history) lives outside React and
 * is read by canvases on each animation frame; React state holds what the
 * screens show as text.
 */
export interface RoomLive {
  frame: FrameMsg | null;
  lastPresent: FrameMsg | null;
  /** last 60 s of fall score, for the room chart */
  fall: { t: number; v: number }[];
  /** last 60 s of posture, for the ribbon under each camera */
  postures: { t: number; posture: Posture }[];
}

export const live: Record<string, RoomLive> = {};
const HISTORY_S = 60;

export function roomLive(room: string): RoomLive {
  return (live[room] ??= { frame: null, lastPresent: null, fall: [], postures: [] });
}

interface Store {
  hello: Hello | null;
  connected: boolean;
  /** server clock minus this device's clock, seconds */
  offset: number;
  incidents: Record<string, Incident>;
  events: LogEvent[];
  status: Record<string, RoomStatus>;
  scores: Record<string, ScoresMsg>;
  postureSince: Record<string, number>;
  /** bumped on every alert so the wall can react (sound, spotlight) */
  alertTick: number;
  /** the server refused this device's sign-in (wrong or old token) */
  unauthorized: boolean;
}

export const useStore = create<Store>(() => ({
  hello: null,
  connected: false,
  offset: 0,
  incidents: {},
  events: [],
  status: {},
  scores: {},
  postureSince: {},
  alertTick: 0,
  unauthorized: false,
}));

export const serverNow = () => Date.now() / 1000 + useStore.getState().offset;

function trim<T extends { t: number }>(xs: T[], now: number) {
  while (xs.length && xs[0].t < now - HISTORY_S) xs.shift();
}

export function handle(msg: ServerMsg) {
  switch (msg.type) {
    case "hello": {
      const incidents: Record<string, Incident> = {};
      for (const i of msg.incidents) incidents[i.id] = i;
      const status: Record<string, RoomStatus> = {};
      for (const s of msg.status.rooms) status[s.room] = s;
      useStore.setState({
        hello: msg,
        incidents,
        events: msg.events.slice().reverse(),
        status,
        offset: msg.status.now - Date.now() / 1000,
      });
      break;
    }
    case "frame": {
      const r = roomLive(msg.room);
      r.frame = msg;
      if (msg.present) r.lastPresent = msg;
      break;
    }
    case "scores": {
      const r = roomLive(msg.room);
      r.fall.push({ t: msg.t, v: msg.fall_score });
      r.postures.push({ t: msg.t, posture: msg.posture });
      trim(r.fall, msg.t);
      trim(r.postures, msg.t);
      const s = useStore.getState();
      const prev = s.scores[msg.room];
      useStore.setState({
        scores: { ...s.scores, [msg.room]: msg },
        postureSince:
          prev?.posture === msg.posture
            ? s.postureSince
            : { ...s.postureSince, [msg.room]: msg.t },
      });
      break;
    }
    case "incident":
      useStore.setState((s) => ({
        incidents: { ...s.incidents, [msg.incident.id]: msg.incident },
      }));
      break;
    case "event": {
      const { type: _type, ...ev } = msg;
      useStore.setState((s) => ({ events: [ev, ...s.events].slice(0, 200) }));
      break;
    }
    case "status": {
      const status: Record<string, RoomStatus> = {};
      for (const s of msg.rooms) status[s.room] = s;
      useStore.setState({ status, offset: msg.now - Date.now() / 1000 });
      break;
    }
    case "alert":
      useStore.setState((s) => ({ alertTick: s.alertTick + 1 }));
      break;
  }
}

// ------------------------------------------------------------------ socket

let socket: WebSocket | null = null;
let wanted = "all";
let delay = 500;
let retry: ReturnType<typeof setTimeout> | undefined;

/**
 * One live connection per page. A refusal (code 4401: not signed in) is final until
 * reconnect() is called after signing in; any other drop retries with backoff. The
 * page counts as signed in only once the server's hello arrives, so a connection
 * that opens and is then refused never flips the screen.
 */
function open() {
  clearTimeout(retry);
  retry = undefined;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams({ frames: wanted, ...(identity() ?? {}) });
  const ws = new WebSocket(`${proto}://${location.host}/ws?${q}`);
  socket = ws;
  ws.onopen = () => ws.send(JSON.stringify({ type: "subscribe", frames: wanted }));
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data) as ServerMsg;
    if (msg.type === "hello") {
      delay = 500;
      useStore.setState({ connected: true, unauthorized: false });
    }
    handle(msg);
  };
  ws.onclose = (e) => {
    if (socket !== ws) return; // replaced by reconnect()
    socket = null;
    if (e.code === 4401) {
      useStore.setState({ connected: false, unauthorized: true });
      return;
    }
    useStore.setState({ connected: false });
    retry = setTimeout(open, delay);
    delay = Math.min(delay * 2, 5000);
  };
}

export function connect(frames = "all") {
  wanted = frames;
  if (!socket && retry === undefined) open();
}

/** Open a fresh connection now, e.g. with the identity just signed in. */
export function reconnect() {
  clearTimeout(retry);
  retry = undefined;
  delay = 500;
  const old = socket;
  socket = null;
  if (old && old.readyState <= WebSocket.OPEN) old.close();
  open();
}

/** Change which rooms' skeleton frames this page receives. */
export function subscribe(frames: string) {
  wanted = frames;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "subscribe", frames }));
  }
}

// ------------------------------------------------------------------ derived

export const ACTIVE_STATES = new Set(["open", "escalated", "acknowledged", "attending"]);
export const LEVEL_RANK: Record<string, number> = { CHECK: 1, OFFLINE: 2, FALL: 3, "ON FLOOR": 4 };

/** Active incidents, most urgent first: unanswered before answered, then level, then age. */
export function activeIncidents(incidents: Record<string, Incident>) {
  const unanswered = (i: Incident) => (i.state === "open" || i.state === "escalated" ? 1 : 0);
  return Object.values(incidents)
    .filter((i) => ACTIVE_STATES.has(i.state))
    .sort(
      (a, b) =>
        unanswered(b) - unanswered(a) ||
        LEVEL_RANK[b.level] - LEVEL_RANK[a.level] ||
        a.opened_at - b.opened_at,
    );
}
