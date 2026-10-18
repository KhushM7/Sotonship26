import { useEffect, useRef } from "react";
import type { Keypoint, Posture } from "../contracts";
import { drawSkeleton, fit, sizeCanvas } from "../skeleton";
import { live, serverNow, useStore } from "../store";

export type Tone = "calm" | "alarm" | "caution" | "muted";

export const TONE: Record<Tone, string> = {
  calm: "#dcdde1",
  alarm: "#e0575c",
  caution: "#d6a243",
  muted: "#8b8e96",
};

function useAnimationFrame(draw: () => void, deps: unknown[]) {
  useEffect(() => {
    let raf = 0;
    const loop = () => {
      draw();
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, deps);
}

function lineWidth(boxH: number) {
  return Math.max(1.5, Math.min(4, boxH / 120));
}

/** The room as the sensor sees it: a skeleton on dark, never the camera image. */
export function LiveSkeleton({ room, tone = "calm", className = "" }: { room: string; tone?: Tone; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useAnimationFrame(() => {
    const c = ref.current;
    if (!c) return;
    const { ctx, w, h } = sizeCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const r = live[room];
    const aspect = useStore.getState().status[room]?.aspect ?? 16 / 9;
    const box = fit(w, h, aspect);
    // the edge of the camera's view, as a hairline
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
    if (!r) return;
    const lw = lineWidth(box.h);
    if (r.frame?.present) {
      drawSkeleton(ctx, r.frame.kp, box, TONE[tone], 1, lw);
    } else if (r.lastPresent) {
      // last seen: a fading ghost where the person was
      const gone = serverNow() - r.lastPresent.t;
      const a = Math.max(0.12, 0.45 - gone / 40);
      ctx.setLineDash([3, 4]);
      drawSkeleton(ctx, r.lastPresent.kp, box, TONE.muted, a, lw);
      ctx.setLineDash([]);
    }
    if (!r.frame?.present) {
      ctx.font = "11px 'Geist Mono Variable', monospace";
      ctx.textAlign = "center";
      ctx.fillStyle = TONE.muted;
      ctx.fillText("OUT OF VIEW", w / 2, h / 2);
    }
  }, [room, tone]);
  return <canvas ref={ref} className={`block h-full w-full ${className}`} />;
}

type ReplayFrame = { t: number; kp: Keypoint[]; present: boolean };

/** Loops the seconds before an alert. */
export function ReplaySkeleton({ frames, aspect, fps, className = "" }: { frames: ReplayFrame[]; aspect: number; fps: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useAnimationFrame(() => {
    const c = ref.current;
    if (!c || !frames.length) return;
    const { ctx, w, h } = sizeCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const box = fit(w, h, aspect);
    const loopS = frames.length / fps + 1.2; // a short pause before it repeats
    const i = Math.floor(((performance.now() / 1000) % loopS) * fps);
    const f = frames[Math.min(i, frames.length - 1)];
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
    if (f.present) drawSkeleton(ctx, f.kp, box, TONE.calm, 0.9, lineWidth(box.h));
    // progress hairline along the bottom
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.fillRect(box.x, box.y + box.h - 2, (box.w * Math.min(i, frames.length)) / frames.length, 2);
  }, [frames, aspect, fps]);
  return <canvas ref={ref} className={`block h-full w-full ${className}`} />;
}

const RIBBON: Record<Posture, string> = {
  upright: "#3a3d44",
  low_on_purpose: "#6b6e76",
  falling: TONE.alarm,
  on_floor: TONE.alarm,
  not_visible: "transparent",
};

/** The last 60 s of posture as a thin strip: grey for normal, red when on the floor. */
export function PostureRibbon({ room, className = "" }: { room: string; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useAnimationFrame(() => {
    const c = ref.current;
    if (!c) return;
    const { ctx, w, h } = sizeCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const ps = live[room]?.postures ?? [];
    const now = serverNow();
    for (let i = 0; i < ps.length; i++) {
      const t0 = ps[i].t;
      const t1 = i + 1 < ps.length ? ps[i + 1].t : now;
      const x0 = w * (1 - (now - t0) / 60);
      const x1 = w * (1 - (now - t1) / 60);
      ctx.fillStyle = RIBBON[ps[i].posture];
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    }
  }, [room]);
  return <canvas ref={ref} className={`block w-full ${className}`} />;
}

/** Fall score over the last 60 s against the backend's thresholds. */
export function FallChart({ room, tHigh, tLow, className = "" }: { room: string; tHigh: number; tLow: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useAnimationFrame(() => {
    const c = ref.current;
    if (!c) return;
    const { ctx, w, h } = sizeCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const pad = 4;
    const y = (v: number) => pad + (h - 2 * pad) * (1 - v);
    const now = serverNow();
    const x = (t: number) => w * (1 - (now - t) / 60);

    ctx.font = "10px 'Geist Mono Variable', monospace";
    ctx.textBaseline = "bottom";
    const lines = tLow === tHigh ? [[tHigh, "FALL"]] : [[tLow, "CHECK"], [tHigh, "FALL"]];
    for (const [v, label] of lines as [number, string][]) {
      ctx.strokeStyle = label === "FALL" ? "rgba(224,87,92,0.55)" : "rgba(214,162,67,0.55)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y(v));
      ctx.lineTo(w, y(v));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText(`${label} ${v.toFixed(2)}`, 4, y(v) - 2);
    }
    const pts = live[room]?.fall ?? [];
    if (pts.length < 2) return;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#cdced3";
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(x(p.t), y(p.v)) : ctx.moveTo(x(p.t), y(p.v))));
    ctx.stroke();
  }, [room, tHigh, tLow]);
  return <canvas ref={ref} className={`block w-full ${className}`} />;
}
