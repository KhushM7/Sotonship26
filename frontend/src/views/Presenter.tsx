import { ArrowCounterClockwiseIcon, PlayIcon, PlugsIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { api, PRESENTER } from "../api";
import { POSTURE_LABEL } from "../format";
import { connect, useStore } from "../store";
import { Button, Label } from "../ui/bits";
import { RoleSignIn } from "../ui/PinPad";

// "live" plays the newest scripts/09_live.py --record session
const SCENARIO_LABEL: Record<string, string> = { live: "webcam recording" };

/** /presenter – demo controls; the server only accepts them with the presenter PIN. */
export function Presenter() {
  const hello = useStore((s) => s.hello);
  const status = useStore((s) => s.status);
  const scores = useStore((s) => s.scores);
  const [msg, setMsg] = useState<string | null>(null);
  const unauthorized = useStore((s) => s.unauthorized);
  useEffect(() => connect("none"), []);

  const run = (label: string, fn: () => Promise<unknown>) => async () => {
    try {
      await fn();
      setMsg(label);
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  if (unauthorized) return <RoleSignIn name={PRESENTER} title="Demo controls" />;
  if (!hello) return null;
  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <Label>Presenter</Label>
      <h1 className="mt-2 text-2xl font-medium tracking-tight">Demo controls</h1>
      <p className="mt-2 max-w-[60ch] text-sm leading-relaxed text-ink-400">
        Scenarios replay recorded clips over their videos; their scores and alerts are the backend's own ({hello.model.run}).
        "webcam recording" plays the newest 09_live.py --record session with the alerts the live script raised. Phones:{" "}
        <span className="font-mono text-ink-200">{hello.public_url}/phone</span>
      </p>

      <ul className="mt-8 divide-y divide-ink-800 border-y border-ink-800">
        {hello.rooms.map((r) => {
          const s = status[r.id];
          return (
            <li key={r.id} className="flex flex-wrap items-center justify-between gap-3 py-4">
              <div className="min-w-0">
                <div className="font-mono text-[11px] uppercase tracking-[0.08em] text-ink-400">{r.camera}</div>
                <div>
                  {r.name}{" "}
                  <span className="text-sm text-ink-500">
                    · {s?.online === false ? "offline" : scores[r.id] ? POSTURE_LABEL[scores[r.id].posture] : "starting"}
                    {s?.scenario ? ` · playing ${s.scenario}` : ""}
                  </span>
                </div>
              </div>
              <div className="flex gap-2">
                {r.scenarios.map((name) => (
                  <Button key={name} kind="ghost" onClick={run(`${r.name}: ${SCENARIO_LABEL[name] ?? name}`, () => api.demo.scenario(r.id, name))}>
                    <PlayIcon /> {SCENARIO_LABEL[name] ?? name}
                  </Button>
                ))}
                <Button kind="ghost" onClick={run(`${r.name}: camera ${s?.online === false ? "on" : "off"}`, () => api.demo.offline(r.id, s?.online !== false))}>
                  <PlugsIcon /> {s?.online === false ? "Reconnect" : "Cut camera"}
                </Button>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="mt-8 flex items-center gap-4">
        <Button kind="ghost" onClick={run("Incidents cleared", api.demo.reset)}>
          <ArrowCounterClockwiseIcon /> Reset incidents
        </Button>
        {msg && <span className="text-sm text-ink-400">{msg}</span>}
      </div>
    </div>
  );
}
