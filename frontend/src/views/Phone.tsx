import { ArrowLeftIcon, BellIcon, BellSlashIcon, CaretRightIcon, SignOutIcon, XIcon } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { adoptLinkIdentity, api, identity, setIdentity, type Identity } from "../api";
import type { Incident, PushMsg } from "../contracts";
import { clock, levelText, incidentReason, STATE_LABEL, unanswered, urgent } from "../format";
import { activeIncidents, connect, reconnect, subscribe, useStore } from "../store";
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
      {me && hello?.demo && <DemoNotification carer={me.carer} />}
    </div>
  );
}

/**
 * Public demo: what ntfy would show on this carer's phone, as a banner on the page.
 * Its buttons do what the real notification's do.
 */
function DemoNotification({ carer }: { carer: string }) {
  const push = useStore((s) => s.lastPush);
  const [shown, setShown] = useState<PushMsg | null>(null);
  useEffect(() => {
    if (!push?.carers.includes(carer)) return;
    setShown(push);
    navigator.vibrate?.(push.priority >= 5 ? [200, 100, 200] : 150);
  }, [push, carer]);
  useEffect(() => {
    if (!shown) return;
    const id = setTimeout(() => setShown(null), 15000);
    return () => clearTimeout(id);
  }, [shown]);
  if (!shown) return null;

  const page = shown.incident ? `/phone/i/${shown.incident}` : null;
  return (
    <div className="fixed inset-x-3 top-3 z-30 animate-rise rounded-2xl border border-ink-700 bg-ink-800/95 p-4 shadow-2xl backdrop-blur">
      <div className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.08em] text-ink-400">
        <span className="flex items-center gap-1.5">
          <BellIcon size={12} /> ntfy · now
        </span>
        <button type="button" onClick={() => setShown(null)} aria-label="Dismiss" className="hover:text-ink-100">
          <XIcon size={14} />
        </button>
      </div>
      <a href={page ?? undefined} className="mt-2 block">
        <div className={`font-medium ${shown.priority >= 5 ? "text-alarm" : "text-ink-100"}`}>{shown.title}</div>
        <div className="mt-0.5 text-sm leading-snug text-ink-300">{shown.message}</div>
      </a>
      {page && shown.actions && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          <a href={`${page}?ack=1`} className="grid h-10 place-items-center rounded-md bg-ink-100 text-sm font-medium text-ink-950">
            I'm going
          </a>
          <a href={page} className="grid h-10 place-items-center rounded-md border border-ink-600 text-sm text-ink-200">
            Open
          </a>
        </div>
      )}
      <p className="mt-3 text-[11px] text-ink-500">Demo: a real shift sends this to the carer's phone through ntfy.</p>
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
  // a public demo has no PINs: the server takes the name as it is
  const demoSignIn = (carer: string) => {
    const id = { carer, token: "" };
    useStore.setState({ unauthorized: false });
    setIdentity(id);
    onDone(id);
    reconnect();
  };

  return (
    <div className="flex flex-1 flex-col">
      <Label className="mt-6">{info?.site.ward}</Label>
      <h1 className="mt-2 text-3xl font-medium tracking-tight">Start your shift</h1>
      {error && <p className="mt-4 text-sm text-alarm">{error}</p>}
      {info?.demo && <p className="mt-3 text-sm text-ink-400">Demo: pick a carer, no PIN needed. Each room alerts its own carers first.</p>}
      {!name ? (
        <div className="mt-8 flex flex-col divide-y divide-ink-800 border-y border-ink-800">
          {(info?.carers ?? []).map((c) => (
            <button key={c.name} type="button" onClick={() => (info?.demo ? demoSignIn(c.name) : setName(c.name))} className="flex min-h-14 items-center justify-between text-left active:bg-ink-900">
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
          {hello.demo
            ? "Demo: alerts show on this screen; a real shift gets them through the ntfy app"
            : carer?.push
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
              const result = r.results[0]?.[1];
              setTest(
                result === "sent"
                  ? "Sent. Check your notifications."
                  : result === "demo"
                    ? "Sent (shown above)."
                    : "Not sent: no topic linked (dry run)",
              );
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
