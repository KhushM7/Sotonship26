import type { Incident } from "./contracts";

/** Who this browser acts as. On the demo laptop the server treats requests as the nurse station. */
export interface Identity {
  carer: string;
  token: string;
}

const KEY = "fallsense.identity";

export function identity(): Identity | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Identity) : null;
  } catch {
    return null;
  }
}

export function setIdentity(id: Identity | null) {
  try {
    if (id) localStorage.setItem(KEY, JSON.stringify(id));
    else localStorage.removeItem(KEY);
  } catch {
    // private mode: the identity lasts for this page only
  }
}

/** A notification link carries ?carer=&token=; keep it so the carer stays signed in. */
export function adoptLinkIdentity() {
  const q = new URLSearchParams(location.search);
  const carer = q.get("carer");
  const token = q.get("token");
  if (carer && token) {
    setIdentity({ carer, token });
    history.replaceState(null, "", location.pathname);
  }
}

async function post<T>(url: string, body: object = {}): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...identity(), ...body }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail ?? `${r.status} ${r.statusText}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  login: (carer: string, pin: string) =>
    post<Identity>("/api/login", { carer, pin }),
  ack: (id: string) => post<Incident>(`/api/incidents/${id}/ack`),
  attend: (id: string) => post<Incident>(`/api/incidents/${id}/attend`),
  escalate: (id: string) => post<Incident>(`/api/incidents/${id}/escalate`),
  dismiss: (id: string) => post<Incident>(`/api/incidents/${id}/dismiss`),
  resolve: (
    id: string,
    body: { outcome: string; injury?: string; reason?: string; note?: string },
  ) => post<Incident>(`/api/incidents/${id}/resolve`, body),
  carerInRoom: (room: string, on: boolean) =>
    post(`/api/rooms/${room}/carer_in_room`, { on }),
  testPush: (carer: string) =>
    post<{ results: [string, string][] }>(`/api/carers/${encodeURIComponent(carer)}/test`),
  replay: async (id: string) => {
    const r = await fetch(`/api/incidents/${id}/replay`);
    return (await r.json()) as {
      fps: number;
      frames: { t: number; kp: ([number, number, number] | null)[]; present: boolean }[];
    };
  },
  demo: {
    scenario: (room: string, name: string) =>
      post(`/api/demo/rooms/${room}/scenario/${name}`),
    offline: (room: string, on: boolean) => post(`/api/demo/rooms/${room}/offline`, { on }),
    reset: () => post("/api/demo/reset"),
  },
};
