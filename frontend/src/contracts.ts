// Contract 6 (live stream) and Contract 7 (incident), mirroring src/fallsense_server.

export type Level = "CHECK" | "FALL" | "ON FLOOR" | "OFFLINE";
export type Posture =
  | "upright"
  | "falling"
  | "on_floor"
  | "low_on_purpose"
  | "not_visible";
export type IncidentState =
  | "open"
  | "escalated"
  | "acknowledged"
  | "attending"
  | "resolved";

/** 13 keypoints: nose, shoulders, elbows, wrists, hips, knees, ankles. */
export type Keypoint = [x: number, y: number, visibility: number] | null;

export interface Room {
  id: string;
  camera: string;
  name: string;
  resident: string;
  carers: string[];
  source: string;
  scenarios: string[];
}

export interface Carer {
  name: string;
  tier: number;
  role: string;
  push: boolean;
}

export interface Incident {
  id: string;
  room: string;
  kind: "fall" | "offline";
  level: Level;
  opened_at: number;
  state: IncidentState;
  tier: number;
  deadline: number | null;
  levels: { t: number; level: Level; score: number; reason: string }[];
  notified: string[];
  acknowledged_by: string | null;
  acknowledged_at: number | null;
  attending_by: string | null;
  attending_at: number | null;
  resolved_by: string | null;
  resolved_at: number | null;
  outcome: string | null;
  injury: string | null;
  false_alarm_reason: string | null;
  note: string | null;
  on_floor_since: number | null;
  floor_ended_at: number | null;
}

export interface LogEvent {
  t: number;
  kind: string;
  room: string;
  text: string;
  incident: string | null;
}

export interface RoomStatus {
  room: string;
  online: boolean;
  source: string;
  clip: string | null;
  /** the clip's source video (see mediaUrl); null = skeleton only */
  video: string | null;
  scenario: string | null;
  aspect: number;
  fps: number;
  carer_in_room_until: number | null;
}

export interface StatusMsg {
  type: "status";
  now: number;
  rooms: RoomStatus[];
}

export interface Hello {
  type: "hello";
  schema_version: number;
  site: { name: string; ward: string };
  /** public demo: no sign-in, demo controls on the wall, alerts go to the on-page phone */
  demo: boolean;
  model: { run: string; t_high: number; t_low: number; on_floor_s: number };
  fps: number;
  bones: [number, number][];
  rooms: Room[];
  carers: Carer[];
  incidents: Incident[];
  events: LogEvent[];
  status: StatusMsg;
  public_url: string;
}

export interface FrameMsg {
  type: "frame";
  room: string;
  t: number;
  kp: Keypoint[];
  present: boolean;
  /** the room's video and the seconds into it at this frame */
  video?: string | null;
  ct?: number;
}

export interface ScoresMsg {
  type: "scores";
  room: string;
  t: number;
  p: [number, number, number, number];
  fall_score: number;
  posture: Posture;
  held: boolean;
}

export interface AlertMsg {
  type: "alert";
  room: string;
  t: number;
  level: Level;
  score: number;
  reason: string;
  incident: string | null;
}

/** Public demo only: the notification a carer's phone would get from ntfy. */
export interface PushMsg {
  type: "push";
  t: number;
  carers: string[];
  priority: number;
  title: string;
  message: string;
  incident: string | null;
  /** "I'm going" / "Open" buttons */
  actions: boolean;
}

export type ServerMsg =
  | Hello
  | FrameMsg
  | ScoresMsg
  | AlertMsg
  | StatusMsg
  | PushMsg
  | { type: "incident"; incident: Incident }
  | ({ type: "event" } & LogEvent);
