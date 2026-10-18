import { ArrowRightIcon, CheckIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { api } from "../api";
import type { Incident } from "../contracts";
import {
  clock,
  firstName,
  incidentReason,
  levelText,
  OUTCOME_LABEL,
  STATE_LABEL,
  unanswered,
  urgent,
} from "../format";
import { useStore } from "../store";
import { Button, Label, LevelMark, Lifecycle, useNow } from "./bits";
import { LiveSkeleton, ReplaySkeleton } from "./canvases";

const NEXT_TIER = { 1: "everyone on shift", 2: "the senior on call", 3: "the senior on call" } as Record<number, string>;

/** The big number: time on the floor if known, else time since the alert. */
export function IncidentTimer({ inc, big = false }: { inc: Incident; big?: boolean }) {
  const now = useNow(250);
  let label = "Since alert";
  let secs = (inc.resolved_at ?? now) - inc.opened_at;
  if (inc.kind === "offline") label = "Offline for";
  else if (inc.on_floor_since !== null) {
    label = inc.floor_ended_at === null ? "On the floor" : "Was on the floor";
    secs = (inc.floor_ended_at ?? now) - inc.on_floor_since;
  }
  return (
    <div>
      <Label>{label}</Label>
      <div className={`font-mono tabular-nums tracking-tight ${big ? "text-6xl" : "text-3xl"} ${urgent(inc) ? "text-alarm" : "text-ink-100"}`}>
        {clock(secs)}
      </div>
    </div>
  );
}

function Deadline({ inc }: { inc: Incident }) {
  const now = useNow(250);
  if (inc.deadline === null || inc.state === "resolved") return null;
  const left = clock(inc.deadline - now);
  const text = unanswered(inc)
    ? `Escalates to ${NEXT_TIER[inc.tier]} in ${left}`
    : `${firstName(inc.acknowledged_by)} expected in the room within ${left}`;
  return <div className="font-mono text-xs text-ink-400">{text}</div>;
}

export function Replay({ inc, className = "" }: { inc: Incident; className?: string }) {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.replay>> | null>(null);
  const aspect = useStore((s) => s.status[inc.room]?.aspect ?? 16 / 9);
  useEffect(() => {
    let alive = true;
    api.replay(inc.id).then((d) => alive && setData(d)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [inc.id]);
  if (!data?.frames.length) {
    return <div className={`grid place-items-center text-xs text-ink-500 ${className}`}>No replay</div>;
  }
  return <ReplaySkeleton frames={data.frames} fps={data.fps} aspect={aspect} className={className} />;
}

/** Everything a carer needs for one incident, with one obvious next step. */
export function IncidentView({ inc, layout }: { inc: Incident; layout: "wall" | "phone" }) {
  const room = useStore((s) => s.hello?.rooms.find((r) => r.id === inc.room));
  const [resolving, setResolving] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const phone = layout === "phone";

  const act = (fn: () => Promise<unknown>) => async () => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const reason = incidentReason(inc);
  const tone = !unanswered(inc) ? "calm" : inc.level === "CHECK" ? "caution" : "alarm";

  return (
    <div className="flex flex-col gap-5">
      <header className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <LevelMark level={inc.level} quiet={inc.state === "resolved"} />
          <h2 className={`mt-2 font-medium tracking-tight ${phone ? "text-3xl" : "text-xl"}`}>
            {room?.name ?? inc.room}
            {room?.resident && inc.kind === "fall" && <span className="text-ink-400"> · {room.resident}</span>}
          </h2>
          <p className="mt-1 text-ink-300">
            {inc.state === "resolved" ? `Resolved by ${inc.resolved_by}` : levelText(inc.level, reason)}
          </p>
        </div>
        <IncidentTimer inc={inc} big={phone} />
      </header>

      {inc.kind === "fall" && (
        <div className={`grid grid-cols-2 gap-px overflow-hidden rounded-md border border-ink-800 bg-ink-800 ${phone ? "h-56" : "h-36"}`}>
          <figure className="relative bg-ink-950">
            <LiveSkeleton room={inc.room} tone={tone} />
            <figcaption className="absolute left-2 top-1.5 font-mono text-[10px] uppercase tracking-[0.08em] text-ink-400">Now</figcaption>
          </figure>
          <figure className="relative bg-ink-950">
            <Replay inc={inc} className="h-full w-full" />
            <figcaption className="absolute left-2 top-1.5 font-mono text-[10px] uppercase tracking-[0.08em] text-ink-400">Before the alert</figcaption>
          </figure>
        </div>
      )}

      <Lifecycle inc={inc} />

      <div className="flex flex-col gap-1">
        <div className="text-sm text-ink-200">
          {inc.state === "resolved"
            ? `${OUTCOME_LABEL[inc.outcome ?? ""] ?? "Resolved"}${inc.false_alarm_reason ? ` · ${inc.false_alarm_reason.replaceAll("_", " ")}` : ""}${inc.injury ? ` · injury: ${inc.injury}` : ""}`
            : inc.state === "acknowledged"
              ? `${inc.acknowledged_by} is on the way`
              : inc.state === "attending"
                ? `${inc.attending_by} is with the resident`
                : STATE_LABEL[inc.state]}
        </div>
        <Deadline inc={inc} />
      </div>

      {error && <p className="text-sm text-alarm">{error}</p>}

      {inc.state !== "resolved" && !resolving && (
        <div className={`flex gap-2 ${phone ? "flex-col" : "flex-wrap"}`}>
          {unanswered(inc) && (
            <Button kind={urgent(inc) ? "alarm" : "primary"} size={phone ? "lg" : "md"} disabled={busy} onClick={act(() => api.ack(inc.id))}>
              I'm going <ArrowRightIcon weight="bold" />
            </Button>
          )}
          {inc.state === "acknowledged" && inc.kind === "fall" && (
            <Button size={phone ? "lg" : "md"} disabled={busy} onClick={act(() => api.attend(inc.id))}>
              I'm with them
            </Button>
          )}
          {inc.state === "attending" && (
            <Button size={phone ? "lg" : "md"} onClick={() => setResolving(true)}>
              <CheckIcon weight="bold" /> Resolve
            </Button>
          )}
          {inc.level === "CHECK" && (
            <Button kind="ghost" size={phone ? "lg" : "md"} disabled={busy} onClick={act(() => api.dismiss(inc.id))}>
              Looks fine, dismiss
            </Button>
          )}
          {inc.kind === "fall" && inc.state !== "attending" && (
            <Button kind="ghost" size={phone ? "lg" : "md"} disabled={busy} onClick={act(() => api.escalate(inc.id))}>
              Can't go, pass it on
            </Button>
          )}
          {!phone && inc.state !== "attending" && inc.kind === "fall" && (
            <Button kind="ghost" disabled={busy} onClick={() => setResolving(true)}>
              Resolve
            </Button>
          )}
        </div>
      )}

      {resolving && <ResolveForm inc={inc} phone={phone} onDone={() => setResolving(false)} />}
    </div>
  );
}

const OUTCOMES = [
  ["real_fall", "Real fall"],
  ["near_miss", "Near miss"],
  ["false_alarm", "False alarm"],
] as const;
const INJURIES = [
  ["none", "No injury"],
  ["minor", "Minor"],
  ["medical_review", "Needs medical review"],
] as const;
const REASONS = [
  ["lying_on_purpose", "Sitting or lying on purpose"],
  ["kneeling_or_bending", "Kneeling or bending"],
  ["exercising", "Exercising"],
  ["carer_helping", "Carer helping"],
  ["someone_else", "Someone else in view"],
  ["other", "Other"],
] as const;

function Choice<T extends string>({ options, value, onChange }: { options: readonly (readonly [T, string])[]; value: T | null; onChange: (v: T) => void }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {options.map(([v, label]) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          className={`min-h-12 rounded-md border px-3 py-2 text-left text-sm transition-colors active:scale-[0.98] ${value === v ? "border-ink-100 bg-ink-100 text-ink-950" : "border-ink-700 text-ink-200 hover:border-ink-500"}`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/** Three taps for the common cases; the answers go back to the backend team as CSV. */
function ResolveForm({ inc, phone, onDone }: { inc: Incident; phone: boolean; onDone: () => void }) {
  const [outcome, setOutcome] = useState<(typeof OUTCOMES)[number][0] | null>(null);
  const [injury, setInjury] = useState<(typeof INJURIES)[number][0] | null>(null);
  const [reason, setReason] = useState<(typeof REASONS)[number][0] | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const ready = outcome && (outcome !== "real_fall" || injury) && (outcome !== "false_alarm" || reason);

  const save = async () => {
    if (!outcome) return;
    try {
      await api.resolve(inc.id, {
        outcome,
        injury: outcome === "real_fall" ? (injury ?? undefined) : undefined,
        reason: outcome === "false_alarm" ? (reason ?? undefined) : undefined,
        note: note || undefined,
      });
      onDone();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="flex animate-rise flex-col gap-4 border-t border-ink-800 pt-4">
      <div className="flex flex-col gap-2">
        <Label>What happened</Label>
        <Choice options={OUTCOMES} value={outcome} onChange={setOutcome} />
      </div>
      {outcome === "real_fall" && (
        <div className="flex flex-col gap-2">
          <Label>Injury</Label>
          <Choice options={INJURIES} value={injury} onChange={setInjury} />
        </div>
      )}
      {outcome === "false_alarm" && (
        <div className="flex flex-col gap-2">
          <Label>What were they doing</Label>
          <Choice options={REASONS} value={reason} onChange={setReason} />
        </div>
      )}
      <div className="flex flex-col gap-2">
        <label htmlFor={`note-${inc.id}`}>
          <Label>Note (optional)</Label>
        </label>
        <textarea
          id={`note-${inc.id}`}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={2}
          className="rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-ink-400"
        />
      </div>
      {error && <p className="text-sm text-alarm">{error}</p>}
      <div className={`flex gap-2 ${phone ? "flex-col" : ""}`}>
        <Button size={phone ? "lg" : "md"} disabled={!ready} onClick={save}>
          Save and close
        </Button>
        <Button kind="ghost" size={phone ? "lg" : "md"} onClick={onDone}>
          Back
        </Button>
      </div>
    </div>
  );
}
