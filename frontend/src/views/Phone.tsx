import { ArrowLeftIcon, BackspaceIcon, BellIcon, BellSlashIcon, CaretRightIcon, SignOutIcon } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { adoptLinkIdentity, api, identity, setIdentity, type Identity } from "../api";
import type { Incident } from "../contracts";
import { clock, levelText, incidentReason, STATE_LABEL, unanswered, urgent } from "../format";
import { activeIncidents, connect, subscribe, useStore } from "../store";
import { Button, Label, LevelMark, useNow } from "../ui/bits";
import { IncidentView } from "../ui/IncidentView";

/** /phone and /phone/i/:id – opened from the ntfy notification on a carer's phone. */
export function Phone({ path }: { path: string }) {
  const [link] = useState(adoptLinkIdentity);
  const [me, setMe] = useState<Identity | null>(identity);
  const hello = useStore((s) => s.hello);
  const incidentId = path.match(/^\/phone\/i\/([^/]+)/)?.[1] ?? null;
  const incident = useStore((s) => (incidentId ? s.incidents[incidentId] : undefined));
  const acked = useRef(false);
  const [ackError, setAckError] = useState<string | null>(null);

  useEffect(() => connect("none"), []);
  // skeleton frames only for the room this carer is looking at
  useEffect(() => subscribe(incident ? incident.room : "none"), [incident?.room]);
  // opened from the notification's "I'm going": acknowledge once the incident is known
  useEffect(() => {
    if (!link.ack || acked.current || !me || !incident || !unanswered(incident)) return;
    acked.current = true;
    api.ack(incident.id).catch((e: Error) => setAckError(e.message));
  }, [link.ack, me, incident]);

  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-lg flex-col px-4 pb-10 pt-[max(16px,env(safe-area-inset-top))]">
      {!hello ? (
        <div className="grid flex-1 place-items-center text-sm text-ink-400">Connecting</div>
      ) : !me ? (
        <SignIn onDone={setMe} />
      ) : incidentId ? (
        incident ? (
          <>
            <a href="/phone" className="mb-6 inline-flex items-center gap-1.5 text-sm text-ink-400">
              <ArrowLeftIcon /> My shift
            </a>
            {ackError && <p className="mb-4 text-sm text-alarm">Could not mark you as going: {ackError}</p>}
            <IncidentView inc={incident} layout="phone" />
          </>
        ) : (
          <p className="mt-10 text-ink-400">This incident is no longer on the system.</p>
        )
      ) : (
        <Shift
          me={me}
          onSignOut={() => {
            setIdentity(null);
            setMe(null);
          }}
        />
      )}
    </div>
  );
}

function SignIn({ onDone }: { onDone: (id: Identity) => void }) {
  const carers = useStore((s) => s.hello?.carers ?? []);
  const site = useStore((s) => s.hello?.site);
  const [name, setName] = useState<string | null>(null);
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);

  const press = async (d: string) => {
    const next = (pin + d).slice(0, 4);
    setPin(next);
    setError(null);
    if (next.length === 4 && name) {
      try {
        const id = await api.login(name, next);
        setIdentity(id);
        onDone(id);
      } catch {
        setError("That PIN is not right");
        setPin("");
      }
    }
  };

  return (
    <div className="flex flex-1 flex-col">
      <Label className="mt-6">{site?.ward}</Label>
      <h1 className="mt-2 text-3xl font-medium tracking-tight">Start your shift</h1>
      {!name ? (
        <div className="mt-8 flex flex-col divide-y divide-ink-800 border-y border-ink-800">
          {carers.map((c) => (
            <button key={c.name} type="button" onClick={() => setName(c.name)} className="flex min-h-14 items-center justify-between text-left active:bg-ink-900">
              <span>
                <span className="block">{c.name}</span>
                <span className="text-xs text-ink-400">{c.role}</span>
              </span>
              <CaretRightIcon className="text-ink-500" />
            </button>
          ))}
        </div>
      ) : (
        <div className="mt-8 flex flex-col items-center">
          <button type="button" onClick={() => { setName(null); setPin(""); }} className="self-start text-sm text-ink-400">
            <ArrowLeftIcon className="inline" /> {name}
          </button>
          <div className="mt-8 flex gap-4" aria-label={`${pin.length} of 4 digits`}>
            {[0, 1, 2, 3].map((i) => (
              <span key={i} className={`size-3 rounded-full border ${i < pin.length ? "border-ink-100 bg-ink-100" : "border-ink-500"}`} />
            ))}
          </div>
          <p className="mt-3 h-5 text-sm text-alarm">{error}</p>
          <div className="mt-4 grid w-full max-w-xs grid-cols-3 gap-3">
            {["1", "2", "3", "4", "5", "6", "7", "8", "9", "", "0", "del"].map((k) =>
              k === "" ? (
                <span key="blank" />
              ) : (
                <button
                  key={k}
                  type="button"
                  onClick={() => (k === "del" ? setPin((p) => p.slice(0, -1)) : press(k))}
                  className="grid h-16 place-items-center rounded-md border border-ink-800 font-mono text-xl active:scale-[0.97] active:bg-ink-900"
                  aria-label={k === "del" ? "Delete" : k}
                >
                  {k === "del" ? <BackspaceIcon size={20} /> : k}
                </button>
              ),
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Shift({ me, onSignOut }: { me: Identity; onSignOut: () => void }) {
  const hello = useStore((s) => s.hello)!;
  const incidents = useStore((s) => s.incidents);
  const active = useMemo(() => activeIncidents(incidents), [incidents]);
  const carer = hello.carers.find((c) => c.name === me.carer);
  const mine = (i: Incident) =>
    i.notified.includes(me.carer) || i.acknowledged_by === me.carer || i.attending_by === me.carer;
  const [test, setTest] = useState<string | null>(null);

  return (
    <div className="flex flex-col">
      <div className="mt-6 flex items-start justify-between">
        <div>
          <Label>On shift</Label>
          <h1 className="mt-2 text-2xl font-medium tracking-tight">{me.carer}</h1>
        </div>
        <button type="button" onClick={onSignOut} className="text-ink-400" aria-label="Sign out">
          <SignOutIcon size={20} />
        </button>
      </div>

      <section className="mt-8">
        <Label className="mb-3">Open incidents</Label>
        {active.length === 0 ? (
          <p className="border-y border-ink-800 py-6 text-ink-300">All quiet. You will get a notification if anyone needs you.</p>
        ) : (
          <ul className="flex flex-col divide-y divide-ink-800 border-y border-ink-800">
            {active.map((i) => (
              <IncidentRow key={i.id} inc={i} mine={mine(i)} />
            ))}
          </ul>
        )}
      </section>

      <section className="mt-10 flex flex-col gap-3">
        <Label>Alerts on this phone</Label>
        <div className="flex items-center gap-2 text-sm text-ink-300">
          {carer?.push ? <BellIcon /> : <BellSlashIcon />}
          {carer?.push
            ? "Linked to your ntfy topic"
            : "No ntfy topic linked yet: ask the shift lead to add one"}
        </div>
        <Button
          kind="ghost"
          size="lg"
          onClick={async () => {
            setTest("Sending");
            try {
              const r = await api.testPush(me.carer);
              setTest(r.results[0]?.[1] === "sent" ? "Sent. Check your notifications." : "Not sent: no topic linked (dry run)");
            } catch (e) {
              setTest((e as Error).message);
            }
          }}
        >
          Send me a test alert
        </Button>
        {test && <p className="text-sm text-ink-400">{test}</p>}
      </section>
    </div>
  );
}

function IncidentRow({ inc, mine }: { inc: Incident; mine: boolean }) {
  const room = useStore((s) => s.hello?.rooms.find((r) => r.id === inc.room));
  const now = useNow(1000);
  return (
    <li>
      <a href={`/phone/i/${inc.id}`} className="flex min-h-16 items-center justify-between gap-4 py-3 active:bg-ink-900">
        <div className="min-w-0">
          <LevelMark level={inc.level} quiet={!unanswered(inc)} />
          <div className="mt-1 truncate">
            {room?.name} <span className="text-ink-500">· {levelText(inc.level, incidentReason(inc))}</span>
          </div>
          <div className="text-xs text-ink-400">
            {STATE_LABEL[inc.state]}
            {mine ? "" : " · not assigned to you"}
          </div>
        </div>
        <span className={`font-mono text-lg tabular-nums ${urgent(inc) ? "text-alarm" : "text-ink-300"}`}>{clock(now - inc.opened_at)}</span>
      </a>
    </li>
  );
}
