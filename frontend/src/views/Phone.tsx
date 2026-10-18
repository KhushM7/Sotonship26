import { ArrowLeftIcon, BellIcon, BellSlashIcon, CaretRightIcon, SignOutIcon } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { adoptLinkIdentity, api, identity, setIdentity, type Identity } from "../api";
import type { Incident } from "../contracts";
import { clock, levelText, incidentReason, STATE_LABEL, unanswered, urgent } from "../format";
import { activeIncidents, connect, subscribe, useStore } from "../store";
import { Button, Label, LevelMark, useNow } from "../ui/bits";
import { IncidentView } from "../ui/IncidentView";
import { PinPad } from "../ui/PinPad";

/** /phone and /phone/i/:id – opened from the ntfy notification on a carer's phone. */
export function Phone({ path }: { path: string }) {
  const [link] = useState(adoptLinkIdentity);
  const [me, setMe] = useState<Identity | null>(identity);
  const hello = useStore((s) => s.hello);
  const incidentId = path.match(/^\/phone\/i\/([^/]+)/)?.[1] ?? null;
  const incident = useStore((s) => (incidentId ? s.incidents[incidentId] : undefined));
  const acked = useRef(false);
  const [ackError, setAckError] = useState<string | null>(null);
  const unauthorized = useStore((s) => s.unauthorized);

  // PinPad has already stored the identity and reconnected
  const signIn = (id: Identity | null) => {
    if (!id) setIdentity(null);
    setMe(id);
  };
  // the server no longer accepts this phone's sign-in (e.g. the shift was reset)
  useEffect(() => {
    if (unauthorized && me) signIn(null);
  }, [unauthorized, me]);

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
      {!me ? (
        <SignIn onDone={signIn} />
      ) : !hello ? (
        <div className="grid flex-1 place-items-center text-sm text-ink-400">Connecting</div>
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
        <Shift me={me} onSignOut={() => signIn(null)} />
      )}
    </div>
  );
}

function SignIn({ onDone }: { onDone: (id: Identity) => void }) {
  const [info, setInfo] = useState<Awaited<ReturnType<typeof api.publicInfo>> | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.publicInfo().then(setInfo).catch(() => setError("Cannot reach the server"));
  }, []);

  return (
    <div className="flex flex-1 flex-col">
      <Label className="mt-6">{info?.site.ward}</Label>
      <h1 className="mt-2 text-3xl font-medium tracking-tight">Start your shift</h1>
      {error && <p className="mt-4 text-sm text-alarm">{error}</p>}
      {!name ? (
        <div className="mt-8 flex flex-col divide-y divide-ink-800 border-y border-ink-800">
          {(info?.carers ?? []).map((c) => (
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
        <div className="mt-8 flex flex-col">
          <button type="button" onClick={() => setName(null)} className="mb-8 self-start text-sm text-ink-400">
            <ArrowLeftIcon className="inline" /> {name}
          </button>
          <PinPad name={name} onDone={onDone} />
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
