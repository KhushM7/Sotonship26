// Alert tones made with Web Audio, so there are no sound files to load. Browsers
// only allow sound after a click, which is what "Start monitoring" is for.

let ctx: AudioContext | null = null;
let muted = false;

export function unlockSound() {
  ctx ??= new AudioContext();
  void ctx.resume();
}

export function setMuted(m: boolean) {
  muted = m;
}

function tone(freq: number, start: number, dur: number, gain = 0.18, type: OscillatorType = "sine") {
  if (!ctx || muted) return;
  const t0 = ctx.currentTime + start;
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc.connect(g).connect(ctx.destination);
  osc.start(t0);
  osc.stop(t0 + dur + 0.05);
}

/** FALL / ON FLOOR / escalation: an insistent falling two-tone, repeated by the caller. */
export function urgentTone() {
  for (const k of [0, 0.5]) {
    tone(988, k, 0.22, 0.2, "triangle");
    tone(740, k + 0.24, 0.22, 0.2, "triangle");
  }
}

/** CHECK: one soft chime. */
export function checkTone() {
  tone(660, 0, 0.5, 0.12);
  tone(880, 0.18, 0.6, 0.1);
}

/** Camera offline: one low tone. */
export function offlineTone() {
  tone(220, 0, 0.8, 0.16, "sine");
}
