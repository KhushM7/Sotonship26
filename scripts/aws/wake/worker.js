// Cloudflare Worker in front of https://fall-safe.bid (deployed by .\scripts\aws.ps1 wake).
//
// Visitors go straight through to the site (the Cloudflare tunnel, WebSockets included)
// while it is up. When the AWS instance is stopped (scripts/aws/idle-stop.sh stops it
// after a while with nobody watching), the tunnel has no server and Cloudflare answers
// 530: the visitor gets a "wake the demo" page instead, whose button starts the
// instance. Only a button press starts it, so crawlers never do.
//
// Secrets (wrangler secret put): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY of the IAM
// user fallsense-wake, allowed ec2:StartInstances on this one instance and nothing else
// but ec2:DescribeInstances. Vars: INSTANCE_ID, AWS_REGION.

import { AwsClient } from "aws4fetch";

// no server behind the tunnel (530), or the server not answering yet
const DOWN = new Set([502, 503, 504, 520, 521, 522, 523, 524, 530]);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/__wake") {
      if (request.method !== "POST") return text("POST only", 405);
      return json(await ec2(env, "StartInstances"));
    }
    if (url.pathname === "/__wake/state") return json(await ec2(env, "DescribeInstances"));

    let res = null;
    try {
      res = await fetch(request);
    } catch {
      // origin unreachable: treat as asleep
    }
    if (res && !DOWN.has(res.status)) return res;
    if (request.headers.get("Upgrade") || url.pathname.startsWith("/api/")) {
      return text("The demo is asleep", 503);
    }
    return new Response(PAGE, {
      status: 503,
      headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" },
    });
  },
};

/** One EC2 Query API call on our instance; returns {state} (pending, running, stopping, ...). */
async function ec2(env, action) {
  const aws = new AwsClient({
    accessKeyId: env.AWS_ACCESS_KEY_ID,
    secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    region: env.AWS_REGION,
    service: "ec2",
  });
  const body = new URLSearchParams({ Action: action, Version: "2016-11-15", "InstanceId.1": env.INSTANCE_ID });
  const r = await aws.fetch(`https://ec2.${env.AWS_REGION}.amazonaws.com/`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded; charset=utf-8" },
    body: body.toString(),
  });
  const xml = await r.text();
  // just stopped for being idle: it cannot start until it has finished stopping
  if (xml.includes("IncorrectInstanceState")) return { state: "stopping" };
  if (!r.ok) return { state: "error", error: (xml.match(/<Message>([^<]*)/) ?? [])[1] ?? r.status };
  const m = xml.match(/<(?:currentState|instanceState)>\s*<code>\d+<\/code>\s*<name>(\w+)<\/name>/);
  return { state: m ? m[1] : "unknown" };
}

function json(body) {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function text(body, status) {
  return new Response(body, { status, headers: { "Cache-Control": "no-store" } });
}

const PAGE = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fallsense · live demo</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; min-height: 100dvh; display: grid; place-items: center; background: #0e0f11; color: #ececee;
         font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { width: 100%; max-width: 30rem; padding: 0 24px; box-sizing: border-box; }
  .label { font: 500 11px ui-monospace, "SF Mono", Consolas, monospace; letter-spacing: .08em; text-transform: uppercase; color: #8b8e96; }
  h1 { margin: 8px 0 0; font-size: 30px; font-weight: 500; letter-spacing: -.01em; }
  p { color: #a9abb2; margin: 12px 0 0; }
  button { margin-top: 32px; width: 100%; height: 56px; border: 0; border-radius: 6px; background: #ececee; color: #0e0f11;
           font: inherit; font-weight: 500; cursor: pointer; }
  button:hover { background: #fff; }
  button:disabled { background: #2c2e34; color: #8b8e96; cursor: default; }
  #status { min-height: 1.5em; margin-top: 14px; font: 13px ui-monospace, "SF Mono", Consolas, monospace; color: #8b8e96; }
  a { color: #cdced3; }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #cdced3; margin-right: 8px;
         vertical-align: middle; animation: beat 1.6s ease-in-out infinite; }
  @keyframes beat { 50% { opacity: .35; } }
</style>
</head>
<body>
<main>
  <div class="label">fallsense · live demo</div>
  <h1>The demo is asleep</h1>
  <p>fallsense spots falls in care-home rooms from the skeleton alone and alerts the carer's phone.
     The live demo replays real falls through our model, and you can answer the alerts yourself.</p>
  <p>To keep it free to run, its server sleeps when nobody is watching. Waking it takes about a minute.</p>
  <button id="wake" type="button">Wake the demo</button>
  <div id="status"></div>
  <p style="font-size:14px">Code and write-up: <a href="https://github.com/KhushM7/Sotonship26">github.com/KhushM7/Sotonship26</a></p>
</main>
<script>
  const button = document.getElementById("wake");
  const status = document.getElementById("status");
  const STATES = { pending: "Starting the server", running: "Server up, starting the demo", stopping: "Finishing its last nap",
                   stopped: "Asking it to start" };
  const say = (t) => { status.innerHTML = '<span class="dot"></span>' + t; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function call(path, method) {
    try { return await (await fetch(path, { method, cache: "no-store" })).json(); } catch { return { state: "unknown" }; }
  }
  button.onclick = async () => {
    button.disabled = true;
    const started = Date.now();
    let s = await call("/__wake", "POST");
    for (;;) {
      if (s.state === "error") { status.textContent = "Could not wake it: " + s.error; button.disabled = false; return; }
      const secs = Math.round((Date.now() - started) / 1000);
      say((STATES[s.state] || "Waking") + "… " + secs + " s");
      const up = await fetch("/api/public", { cache: "no-store" }).then((r) => r.ok).catch(() => false);
      if (up) { say("Ready"); location.reload(); return; }
      await sleep(3000);
      s = await call(s.state === "stopped" || s.state === "stopping" ? "/__wake" : "/__wake/state", s.state === "stopped" || s.state === "stopping" ? "POST" : "GET");
    }
  };
</script>
</body>
</html>`;
