import { ArrowLeftIcon, DownloadSimpleIcon } from "@phosphor-icons/react";
import { useEffect, useMemo } from "react";
import { clock, OUTCOME_LABEL, STATE_LABEL, timeOfDay } from "../format";
import { api, STATION } from "../api";
import { connect, useStore } from "../store";
import { RoleSignIn } from "../ui/PinPad";
import { Label } from "../ui/bits";

function median(xs: number[]) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

/** /log – the shift handover: what happened, how fast people got there, what was false. */
export function Log() {
  const hello = useStore((s) => s.hello);
  const incidents = useStore((s) => s.incidents);
  const unauthorized = useStore((s) => s.unauthorized);
  useEffect(() => connect("none"), []);

  const rows = useMemo(
    () => Object.values(incidents).sort((a, b) => b.opened_at - a.opened_at),
    [incidents],
  );
  const falls = rows.filter((i) => i.kind === "fall");
  const ackS = median(falls.flatMap((i) => (i.acknowledged_at ? [i.acknowledged_at - i.opened_at] : [])));
  const floorS = median(falls.flatMap((i) => (i.on_floor_since && i.floor_ended_at ? [i.floor_ended_at - i.on_floor_since] : [])));
  const falseAlarms = falls.filter((i) => i.outcome === "false_alarm").length;
  const roomName = (id: string) => hello?.rooms.find((r) => r.id === id)?.name ?? id;

  const stats = [
    ["Incidents", String(falls.length)],
    ["Median time to respond", ackS === null ? "–" : clock(ackS)],
    ["Median time on floor", floorS === null ? "–" : clock(floorS)],
    ["False alarms", `${falseAlarms} of ${falls.filter((i) => i.outcome).length} closed`],
  ];

  if (unauthorized) return <RoleSignIn name={STATION} title="Shift log" />;
  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <a href="/" className="inline-flex items-center gap-1.5 text-sm text-ink-400 hover:text-ink-100">
        <ArrowLeftIcon /> Wall
      </a>
      <div className="mt-6 flex items-end justify-between gap-4">
        <div>
          <Label>{hello?.site.ward}</Label>
          <h1 className="mt-2 text-2xl font-medium tracking-tight">Shift log</h1>
        </div>
        <button type="button" onClick={() => api.exportCsv()} className="inline-flex items-center gap-2 text-sm text-ink-300 hover:text-ink-100">
          <DownloadSimpleIcon /> Export CSV
        </button>
      </div>

      <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-ink-800 bg-ink-800 md:grid-cols-4">
        {stats.map(([k, v]) => (
          <div key={k} className="bg-ink-950 p-5">
            <dt className="text-xs text-ink-400">{k}</dt>
            <dd className="mt-2 font-mono text-2xl tabular-nums tracking-tight">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="mt-8 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-500">
            <tr className="border-b border-ink-800">
              {["Time", "Room", "Level", "Responded", "On floor", "Status", "Outcome"].map((h) => (
                <th key={h} className="py-2 pr-4 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-850">
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="py-8 text-ink-500">No incidents this shift.</td>
              </tr>
            )}
            {rows.map((i) => (
              <tr key={i.id}>
                <td className="py-2.5 pr-4 font-mono text-xs text-ink-400">{timeOfDay(i.opened_at)}</td>
                <td className="pr-4">{roomName(i.room)}</td>
                <td className="pr-4 font-mono text-xs">{i.level}</td>
                <td className="pr-4 text-ink-300">
                  {i.acknowledged_at ? `${clock(i.acknowledged_at - i.opened_at)} · ${i.acknowledged_by}` : "–"}
                </td>
                <td className="pr-4 font-mono text-xs text-ink-300">
                  {i.on_floor_since && i.floor_ended_at ? clock(i.floor_ended_at - i.on_floor_since) : "–"}
                </td>
                <td className="pr-4 text-ink-300">{STATE_LABEL[i.state]}</td>
                <td className="text-ink-300">
                  {i.outcome ? OUTCOME_LABEL[i.outcome] : "–"}
                  {i.false_alarm_reason ? ` · ${i.false_alarm_reason.replaceAll("_", " ")}` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
