import {
  BellIcon,
  BellSlashIcon,
  DeviceMobileIcon,
  HandPalmIcon,
  PlugsIcon,
  ShieldCheckIcon,
  SpeakerHighIcon,
  SpeakerSlashIcon,
  XIcon,
} from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { api } from "../api";
import type { Incident, Room } from "../contracts";
import {
  ago,
  clock,
  firstName,
  incidentReason,
  levelText,
  POSTURE_LABEL,
  timeOfDay,
  unanswered,
  urgent,
} from "../format";
import { checkTone, offlineTone, setMuted, unlockSound, urgentTone } from "../sound";
import { activeIncidents, connect, useStore } from "../store";
import { Button, Label, LevelMark, useNow } from "../ui/bits";
import { FallChart, LiveSkeleton, PostureRibbon, type Tone } from "../ui/canvases";
import { IncidentView } from "../ui/IncidentView";

export function Wall() {
  const hello = useStore((s) => s.hello);
  const incidents = useStore((s) => s.incidents);
  const [started, setStarted] = useState(false);
  const [sound, setSound] = useState(true);
  const [openRoom, setOpenRoom] = useState<string | null>(null);

  useEffect(() => connect("all"), []);
  useAlertSounds(started);
  useEffect(() => setMuted(!sound), [sound]);

  const active = useMemo(() => activeIncidents(incidents), [incidents]);
  const focus = active.find((i) => i.kind === "fall")?.room ?? null;
  const stage = useStageRoom(focus);

  if (!hello) return <Connecting />;

  return (
    <div className="grid h-dvh grid-rows-[auto_1fr] overflow-hidden">
      <Header sound={sound} onSound={() => setSound((s) => !s)} />
      <main className="grid min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_380px]">
        <Cameras rooms={hello.rooms} stage={stage} onOpen={setOpenRoom} />
        <Rail active={active} />
      </main>
      {openRoom && <RoomSheet roomId={openRoom} onClose={() => setOpenRoom(null)} />}
      {!started && (
        <StartOverlay
          onStart={() => {
            unlockSound();
            setStarted(true);
          }}
        />
      )}
    </div>
  );
}

/** The spotlight follows the most urgent fall, with a cross-fade when it moves. */
function useStageRoom(focus: string | null) {
  const [stage, setStage] = useState<string | null>(focus);
  useEffect(() => {
    if (focus === stage) return;
    const swap = () => flushSync(() => setStage(focus));
    if (document.startViewTransition) document.startViewTransition(swap);
    else swap();
  }, [focus, stage]);
  return stage;
}

/** Urgent tone every few seconds while a FALL waits for a carer; one tone for CHECK and offline. */
function useAlertSounds(on: boolean) {
  const incidents = useStore((s) => s.incidents);
  const seen = useRef<Set<string> | null>(null);
  useEffect(() => {
    const all = Object.values(incidents);
    if (seen.current === null) {
      seen.current = new Set(all.map((i) => `${i.id}:${i.level}`));
      return;
    }
    for (const i of all) {
      const key = `${i.id}:${i.level}`;
      if (seen.current.has(key) || i.state === "resolved") continue;
      seen.current.add(key);
      if (!on) continue;
      if (i.level === "CHECK") checkTone();
      else if (i.level === "OFFLINE") offlineTone();
    }
  }, [incidents, on]);
  useEffect(() => {
    if (!on) return;
    const tick = () => {
      if (Object.values(useStore.getState().incidents).some(urgent)) urgentTone();
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, [on]);
}

function Header({ sound, onSound }: { sound: boolean; onSound: () => void }) {
  const hello = useStore((s) => s.hello)!;
  const connected = useStore((s) => s.connected);
  const now = useNow(1000);
  const onShift = hello.carers.length;
  return (
    <header className="flex h-14 items-center gap-6 border-b border-ink-800 px-5">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-sm font-semibold tracking-tight">fallsense</span>
        <span className="hidden text-sm text-ink-400 md:inline">
          {hello.site.name} · {hello.site.ward}
        </span>
      </div>
      <div className="hidden flex-1 items-center justify-center gap-2 text-sm text-ink-400 xl:flex">
        <DeviceMobileIcon size={16} />
        Alerts go straight to carers' phones. Nobody needs to watch this screen.
      </div>
      <div className="ml-auto flex items-center gap-5 text-sm">
        <span className="hidden items-center gap-1.5 text-ink-400 md:flex" title="Only skeletons leave the camera; no video is shown or stored">
          <ShieldCheckIcon size={16} /> Skeleton only
        </span>
        <span className="hidden text-ink-400 sm:inline">{onShift} on shift</span>
        <button type="button" onClick={onSound} className="text-ink-300 hover:text-ink-100" aria-label={sound ? "Mute alert sounds" : "Turn alert sounds on"}>
          {sound ? <SpeakerHighIcon size={18} /> : <SpeakerSlashIcon size={18} />}
        </button>
        <span className="flex items-center gap-2 font-mono tabular-nums text-ink-200">
          <span className={`size-1.5 rounded-full ${connected ? "bg-ink-200" : "animate-beat bg-alarm"}`} title={connected ? "Connected" : "Reconnecting"} />
          {timeOfDay(now)}
        </span>
      </div>
    </header>
  );
}

function Cameras({ rooms, stage, onOpen }: { rooms: Room[]; stage: string | null; onOpen: (id: string) => void }) {
  if (stage) {
    const rest = rooms.filter((r) => r.id !== stage);
    const main = rooms.find((r) => r.id === stage)!;
    return (
      <section className="grid min-h-0 grid-rows-[minmax(0,1fr)_auto] gap-3 p-3">
        <CameraTile room={main} variant="stage" onOpen={onOpen} />
        <div className="grid grid-cols-5 gap-3">
          {rest.map((r) => (
            <CameraTile key={r.id} room={r} variant="strip" onOpen={onOpen} />
          ))}
        </div>
      </section>
    );
  }
  return (
    <section className="grid min-h-0 auto-rows-[minmax(240px,1fr)] grid-cols-1 gap-3 overflow-y-auto p-3 sm:grid-cols-2 xl:grid-cols-3">
      {rooms.map((r) => (
        <CameraTile key={r.id} room={r} variant="grid" onOpen={onOpen} />
      ))}
    </section>
  );
}

function roomIncident(incidents: Record<string, Incident>, room: string) {
  return activeIncidents(incidents).find((i) => i.room === room);
}

function CameraTile({ room, variant, onOpen }: { room: Room; variant: "grid" | "stage" | "strip"; onOpen: (id: string) => void }) {
  const status = useStore((s) => s.status[room.id]);
  const scores = useStore((s) => s.scores[room.id]);
  const since = useStore((s) => s.postureSince[room.id]);
  const inc = useStore((s) => roomIncident(s.incidents, room.id));
  const now = useNow(1000);
  const offline = status ? !status.online : false;
  const fall = inc?.kind === "fall" ? inc : undefined;
  const tone: Tone = fall && unanswered(fall) ? (fall.level === "CHECK" ? "caution" : "alarm") : "calm";
  const muted = (status?.carer_in_room_until ?? 0) > now;
  const small = variant === "strip";
  const waiting = fall && unanswered(fall);
  const ring = fall
    ? waiting
      ? fall.level === "CHECK"
        ? "border-caution"
        : "border-alarm"
      : "border-ink-500"
    : "border-ink-800";

  return (
    <button
      type="button"
      onClick={() => onOpen(room.id)}
      className={`relative flex flex-col overflow-hidden rounded-lg border ${small ? "" : "min-h-0"} bg-ink-900 text-left transition-colors hover:border-ink-500 ${ring}`}
    >
      {waiting && <span className={`pointer-events-none absolute inset-0 z-10 animate-beat rounded-lg border-2 ${fall.level === "CHECK" ? "border-caution" : "border-alarm"}`} />}

      {/* the camera view: skeleton on dark, CCTV caption in the corner */}
      <div className={`relative bg-ink-950 ${small ? "aspect-video shrink-0" : "min-h-0 flex-1"} ${offline ? "hatch" : ""}`}>
        {!offline && <LiveSkeleton room={room.id} tone={tone} className="absolute inset-0" />}
        {offline && (
          <div className="absolute inset-0 grid place-items-center text-center text-ink-300">
            <div className="flex flex-col items-center gap-1">
              <PlugsIcon size={small ? 18 : 26} />
              <div className={small ? "text-xs" : "text-sm"}>Camera offline</div>
              {inc?.kind === "offline" && !small && <div className="font-mono text-xs text-ink-500">since {timeOfDay(inc.opened_at)}</div>}
            </div>
          </div>
        )}
        <div className={`absolute inset-x-0 top-0 flex items-center justify-between font-mono uppercase tracking-[0.08em] text-ink-400 ${small ? "p-2 text-[10px]" : "p-3 text-[11px]"}`}>
          <span>{room.camera}</span>
          {!small && (
            <span className="flex items-center gap-1.5">
              <span className={`size-1.5 rounded-full ${offline ? "bg-ink-500" : "bg-ink-200"}`} />
              {status?.source === "live" ? "Live" : "Replay"}
            </span>
          )}
        </div>
        {variant === "stage" && fall && <StageTimer inc={fall} />}
        {muted && !small && (
          <span className="absolute bottom-3 right-3 flex items-center gap-1 rounded-full border border-ink-700 bg-ink-950/80 px-2 py-0.5 text-[11px] text-ink-300">
            <HandPalmIcon size={12} /> Carer in room
          </span>
        )}
      </div>

      {/* caption: who, and what is happening */}
      <div className={`border-t border-ink-800 ${small ? "px-2 py-1.5" : "px-3 py-2.5"}`}>
        <div className="flex items-baseline justify-between gap-3">
          <div className={`truncate font-medium ${variant === "stage" ? "text-base" : small ? "text-xs" : "text-sm"}`}>
            {room.name}
            {!small && <span className="font-normal text-ink-500"> · {room.resident}</span>}
          </div>
          {fall ? (
            small ? <LevelMark level={fall.level} quiet={!waiting} /> : <TileIncident inc={fall} />
          ) : (
            !offline && scores && !small && (
              <div className="shrink-0 text-sm text-ink-300">
                {POSTURE_LABEL[scores.posture]}
                {since !== undefined && <span className="font-mono text-xs text-ink-500"> · {ago(now - since)}</span>}
              </div>
            )
          )}
        </div>
        {!offline && !small && <PostureRibbon room={room.id} className="mt-2 h-[3px] rounded-full bg-ink-850" />}
      </div>
    </button>
  );
}

function TileIncident({ inc }: { inc: Incident }) {
  return (
    <div className="flex shrink-0 items-baseline gap-2 text-sm text-ink-200">
      <LevelMark level={inc.level} quiet={!unanswered(inc)} />
      <span className="truncate">
        {inc.state === "acknowledged"
          ? `${firstName(inc.acknowledged_by)} on the way`
          : inc.state === "attending"
            ? `${firstName(inc.attending_by)} is with them`
            : levelText(inc.level, incidentReason(inc))}
      </span>
    </div>
  );
}

/** Large timer in the spotlight: time on the floor once known, else since the alert. */
function StageTimer({ inc }: { inc: Incident }) {
  const now = useNow(250);
  const floor = inc.on_floor_since !== null && inc.floor_ended_at === null;
  const secs = floor ? now - inc.on_floor_since! : now - inc.opened_at;
  return (
    <div className="absolute bottom-4 left-4">
      <Label>{floor ? "On the floor" : "Since alert"}</Label>
      <div className={`font-mono text-5xl tabular-nums tracking-tight ${unanswered(inc) ? "text-alarm" : "text-ink-100"}`}>{clock(secs)}</div>
    </div>
  );
}

function Rail({ active }: { active: Incident[] }) {
  return (
    <aside className="hidden min-h-0 flex-col border-l border-ink-800 lg:flex">
      <section className="max-h-[62%] shrink-0 overflow-y-auto border-b border-ink-800 p-5">
        <Label className="mb-4">Needs attention{active.length ? ` · ${active.length}` : ""}</Label>
        {active.length === 0 ? (
          <div className="animate-rise text-sm leading-relaxed text-ink-400">
            <p className="text-ink-200">All quiet.</p>
            <p className="mt-1">When something happens, the carer for that room is alerted on their phone first. It shows here too.</p>
          </div>
        ) : (
          <div className="flex flex-col divide-y divide-ink-800">
            {active.map((i) => (
              <div key={i.id} className="animate-rise py-4 first:pt-0 last:pb-0">
                <IncidentView inc={i} layout="wall" />
              </div>
            ))}
          </div>
        )}
      </section>
      <Carers />
      <Activity />
    </aside>
  );
}

function Carers() {
  const hello = useStore((s) => s.hello)!;
  const incidents = useStore((s) => s.incidents);
  const doing = (name: string) => {
    const i = activeIncidents(incidents).find((x) => x.attending_by === name || x.acknowledged_by === name);
    const room = i && hello.rooms.find((r) => r.id === i.room)?.name;
    if (!i) return "Available";
    return i.state === "attending" ? `With ${room}` : `Going to ${room}`;
  };
  return (
    <section className="shrink-0 border-b border-ink-800 px-5 py-4">
      <Label className="mb-3">On shift</Label>
      <ul className="flex flex-col gap-2 text-sm">
        {hello.carers.map((c) => (
          <li key={c.name} className="flex items-center justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2">
              <span title={c.push ? "Phone alerts on" : "No phone linked: alerts are not sent"} className={c.push ? "text-ink-300" : "text-ink-600"}>
                {c.push ? <BellIcon size={14} /> : <BellSlashIcon size={14} />}
              </span>
              <span className="truncate">{c.name}</span>
              {c.tier === 3 && <span className="text-xs text-ink-500">{c.role}</span>}
            </span>
            <span className="shrink-0 text-xs text-ink-400">{doing(c.name)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Activity() {
  const events = useStore((s) => s.events);
  return (
    <section className="flex min-h-0 flex-1 flex-col px-5 py-4">
      <div className="mb-3 flex items-center justify-between">
        <Label>Activity</Label>
        <a href="/log" className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-500 hover:text-ink-200">
          Shift log
        </a>
      </div>
      <ol className="min-h-0 flex-1 overflow-y-auto text-[13px]">
        {events.length === 0 && <li className="text-ink-500">Nothing yet this shift.</li>}
        {events.map((e, k) => (
          <li key={`${e.t}-${k}`} className="grid grid-cols-[64px_1fr] gap-2 py-1">
            <span className="font-mono text-[11px] leading-5 text-ink-500">{timeOfDay(e.t).slice(0, 5)}</span>
            <span className={e.kind === "alert" || e.kind === "escalate" ? "text-ink-100" : "text-ink-400"}>{e.text}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/** What the sensor sees, for anyone curious how the model decided. */
function RoomSheet({ roomId, onClose }: { roomId: string; onClose: () => void }) {
  const hello = useStore((s) => s.hello)!;
  const room = hello.rooms.find((r) => r.id === roomId)!;
  const status = useStore((s) => s.status[roomId]);
  const scores = useStore((s) => s.scores[roomId]);
  const now = useNow(1000);
  const muted = (status?.carer_in_room_until ?? 0) > now;
  const labels = ["Upright", "Falling", "On the floor", "Sitting or lying"];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-20 flex justify-end bg-ink-950/60" onClick={onClose}>
      <div className="flex h-full w-full max-w-[480px] animate-rise flex-col gap-6 overflow-y-auto border-l border-ink-800 bg-ink-900 p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <div>
            <Label>{room.camera}</Label>
            <h2 className="mt-1 text-xl font-medium tracking-tight">
              {room.name} <span className="text-ink-500">· {room.resident}</span>
            </h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="text-ink-400 hover:text-ink-100">
            <XIcon size={20} />
          </button>
        </div>

        <div className="aspect-video overflow-hidden rounded-md border border-ink-800 bg-ink-950">
          <LiveSkeleton room={roomId} />
        </div>

        <div>
          <Label className="mb-3">What the model sees now</Label>
          <div className="flex flex-col gap-2">
            {labels.map((l, i) => {
              const p = scores?.p[i] ?? 0;
              return (
                <div key={l} className="grid grid-cols-[120px_1fr_44px] items-center gap-3 text-sm">
                  <span className="text-ink-300">{l}</span>
                  <span className="h-1.5 overflow-hidden rounded-full bg-ink-800">
                    <span className={`block h-full rounded-full ${i === 1 || i === 2 ? "bg-ink-100" : "bg-ink-400"}`} style={{ width: `${Math.round(p * 100)}%`, transition: "width 200ms" }} />
                  </span>
                  <span className="text-right font-mono text-xs tabular-nums text-ink-300">{p.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <Label className="mb-2">Fall score, last 60 s</Label>
          <FallChart room={roomId} tHigh={hello.model.t_high} tLow={hello.model.t_low} className="h-28 rounded-md border border-ink-800 bg-ink-950" />
        </div>

        <div className="flex items-center justify-between gap-4 border-t border-ink-800 pt-5">
          <div>
            <div className="text-sm">Carer in room</div>
            <div className="text-xs text-ink-400">{muted ? `Alerts muted for ${clock((status!.carer_in_room_until ?? now) - now)}` : "Mutes alerts for up to 15 min"}</div>
          </div>
          <Button kind={muted ? "primary" : "ghost"} onClick={() => api.carerInRoom(roomId, !muted)}>
            <HandPalmIcon /> {muted ? "End" : "Start"}
          </Button>
        </div>

        <dl className="grid grid-cols-[110px_1fr] gap-y-1.5 border-t border-ink-800 pt-5 font-mono text-xs text-ink-400">
          <dt>Model</dt>
          <dd className="text-ink-200">{hello.model.run}</dd>
          <dt>Thresholds</dt>
          <dd className="text-ink-200">
            FALL {hello.model.t_high} · CHECK {hello.model.t_low} · on floor {hello.model.on_floor_s} s
          </dd>
          <dt>Source</dt>
          <dd className="truncate text-ink-200">
            {status?.source ?? "?"} · {status?.clip ?? "none"}
          </dd>
          <dt>Frames</dt>
          <dd className="text-ink-200">{status?.fps ?? 0} fps</dd>
          {scores?.held && (
            <>
              <dt>Note</dt>
              <dd className="text-ink-200">clip ended; holding its last pose</dd>
            </>
          )}
        </dl>
      </div>
    </div>
  );
}

function StartOverlay({ onStart }: { onStart: () => void }) {
  const hello = useStore((s) => s.hello)!;
  return (
    <div className="fixed inset-0 z-30 grid place-items-center bg-ink-950/85 backdrop-blur-sm">
      <div className="w-full max-w-md animate-rise px-6">
        <Label>{hello.site.ward}</Label>
        <h1 className="mt-2 text-3xl font-medium tracking-tight">{hello.site.name}</h1>
        <p className="mt-3 leading-relaxed text-ink-300">
          {hello.rooms.length} rooms are watched as skeletons only. Falls are sent to carers' phones; this screen plays a tone too.
        </p>
        <Button size="lg" className="mt-8 w-full" onClick={onStart}>
          Start monitoring
        </Button>
      </div>
    </div>
  );
}

function Connecting() {
  return (
    <div className="grid h-dvh place-items-center">
      <div className="flex flex-col items-center gap-3 text-sm text-ink-400">
        <span className="size-1.5 animate-beat rounded-full bg-ink-200" />
        Connecting to the room sensors
      </div>
    </div>
  );
}
