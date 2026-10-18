import { BackspaceIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { api, setIdentity, type Identity } from "../api";
import { reconnect, useStore } from "../store";
import { Label } from "./bits";

/** Four-digit PIN for `name`; signs in, stores the identity and calls onDone. */
export function PinPad({ name, onDone }: { name: string; onDone: (id: Identity) => void }) {
  const [pin, setPin] = useState("");
  const [error, setError] = useState<string | null>(null);

  // physical keyboards work too (the wall and presenter run on laptops)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (/^\d$/.test(e.key)) press(e.key);
      else if (e.key === "Backspace") setPin((p) => p.slice(0, -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const press = async (d: string) => {
    const next = (pin + d).slice(0, 4);
    setPin(next);
    setError(null);
    if (next.length < 4) return;
    try {
      const id = await api.login(name, next);
      useStore.setState({ unauthorized: false });
      setIdentity(id);
      onDone(id);
      reconnect();
    } catch (e) {
      const msg = (e as Error).message;
      setError(msg.includes("too many") ? "Too many wrong PINs. Wait 10 minutes." : "That PIN is not right");
      setPin("");
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div className="flex gap-4" aria-label={`${pin.length} of 4 digits`}>
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
  );
}

/** Full-page sign-in for the wall, log and presenter when the server asks for one. */
export function RoleSignIn({ name, title }: { name: string; title: string }) {
  const [site, setSite] = useState<{ name: string; ward: string } | null>(null);
  useEffect(() => {
    api.publicInfo().then((i) => setSite(i.site)).catch(() => undefined);
  }, []);
  return (
    <div className="grid min-h-dvh place-items-center px-6">
      <div className="w-full max-w-sm animate-rise">
        <Label>{site ? `${site.name} · ${site.ward}` : " "}</Label>
        <h1 className="mt-2 text-3xl font-medium tracking-tight">{title}</h1>
        <p className="mt-2 text-sm text-ink-400">Enter the {name.toLowerCase()} PIN.</p>
        <div className="mt-8">
          <PinPad name={name} onDone={() => undefined} />
        </div>
      </div>
    </div>
  );
}
