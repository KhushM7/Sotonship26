import type { Incident } from "./contracts";

/**
 * Who this page acts as. Each page keeps its own sign-in, so the wall (nurse
 * station), the presenter and a carer's phone can share one browser. With
 * trust_localhost on, pages on the demo laptop need no sign-in at all, and in a
 * public demo no page does (a carer is then just a name, with no token).
 */
export interface Identity {
  carer: string;
  token: string;
  role?: string;
}

export const STATION = "Nurse station";
export const PRESENTER = "Presenter";

const SCOPE = location.pathname.startsWith("/phone")
  ? "phone"
  : location.pathname.startsWith("/presenter")
    ? "presenter"
    : "station";
const KEY = `fallsense.identity.${SCOPE}`;

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

function authHeaders(): Record<string, string> {
  const me = identity();
  return me ? { "X-Carer": me.carer, "X-Token": me.token } : {};
}

/**
 * A notification link carries ?carer=&token= (keep it so the carer stays signed in)
 * and, from the "I'm going" button, &ack=1. The query is then removed so a reload
 * does not acknowledge again.
 */
export function adoptLinkIdentity(): { ack: boolean } {
  const q = new URLSearchParams(location.search);
  const carer = q.get("carer");
  const token = q.get("token");
  const as = q.get("as"); // the demo wall's phone: /phone?as=<carer>
  if (carer && token) setIdentity({ carer, token });
  else if (as) setIdentity({ carer: as, token: "" });
  if (location.search) history.replaceState(null, "", location.pathname);
  return { ack: q.get("ack") === "1" };
}

async function post<T>(url: string, body: object = {}): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ ...identity(), ...body }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail ?? `${r.status} ${r.statusText}`);
  }
  return r.json() as Promise<T>;
}

/** A media address (<video> cannot send headers) carrying this page's sign-in. */
export function mediaUrl(path: string) {
  const me = identity();
  return me ? `${path}?${new URLSearchParams({ carer: me.carer, token: me.token })}` : path;
}

async function get(url: string): Promise<Response> {
  const r = await fetch(url, { headers: authHeaders() });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r;
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
  /** Site and carer names for the sign-in screen; needs no sign-in. */
  publicInfo: async () => {
    const r = await fetch("/api/public");
    return (await r.json()) as {
      site: { name: string; ward: string };
      demo: boolean;
      carers: { name: string; role: string }[];
    };
  },
  replay: async (id: string) =>
    (await (await get(`/api/incidents/${id}/replay`)).json()) as {
      fps: number;
      frames: {
        t: number;
        kp: ([number, number, number] | null)[];
        present: boolean;
        ct?: number;
        video?: string | null;
      }[];
    },
  /** The shift's incidents as a CSV file download. */
  exportCsv: async () => {
    const blob = await (await get("/api/export/incidents.csv")).blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "incidents.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  },
  demo: {
    scenario: (room: string, name: string) =>
      post(`/api/demo/rooms/${room}/scenario/${name}`),
    offline: (room: string, on: boolean) => post(`/api/demo/rooms/${room}/offline`, { on }),
    reset: () => post("/api/demo/reset"),
  },
};
