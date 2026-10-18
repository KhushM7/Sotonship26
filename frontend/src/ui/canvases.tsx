import { useEffect, useRef } from "react";
import { mediaUrl } from "../api";
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

/**
 * Keeps a <video> on the clip time the server's frames carry: it plays freely and
 * is re-seeked only when it drifts, and holds its last frame past the end (a
 * scenario's held tail).
 */
function syncVideo(v: HTMLVideoElement, src: string | null | undefined, t: number | null) {
  const url = src ? mediaUrl(src) : "";
  if (v.dataset.src !== url) {
    v.dataset.src = url;
    if (url) v.src = url;
    else v.removeAttribute("src");
  }
  if (!url || v.readyState < 1) return;
  if (t === null) {
    if (!v.paused) v.pause();
    return;
  }
  const end = v.duration - 0.05;
  if (t >= end) {
    if (!v.paused) v.pause();
    if (Math.abs(v.currentTime - end) > 0.3) v.currentTime = end;
    return;
  }
  if (Math.abs(v.currentTime - t) > 0.3) v.currentTime = t;
  if (v.paused) v.play().catch(() => undefined);
}

/** The source frame's aspect: the video's own once loaded, else the server's. */
function frameAspect(v: HTMLVideoElement | null, fallback: number) {
  return v && v.readyState >= 1 && v.videoHeight ? v.videoWidth / v.videoHeight : fallback;
}

function showing(v: HTMLVideoElement | null) {
  return !!v && !!v.dataset.src && v.readyState >= 2;
}

/** A skeleton drawn over video gets a dark outline so it reads on any background. */
function drawOver(
  ctx: CanvasRenderingContext2D,
  kp: Keypoint[],
  box: { x: number; y: number; w: number; h: number },
  color: string,
  alpha: number,
  lw: number,
  onVideo: boolean,
) {
  if (onVideo) drawSkeleton(ctx, kp, box, "#000", alpha * 0.55, lw + 2.5);
  drawSkeleton(ctx, kp, box, color, alpha, onVideo ? lw + 0.5 : lw);
}

function RoomVideo({ videoRef }: { videoRef: React.RefObject<HTMLVideoElement | null> }) {
  return (
    <video
      ref={videoRef}
      muted
      playsInline
      preload="auto"
      className="absolute inset-0 h-full w-full object-contain opacity-0 transition-opacity duration-300"
      style={{ filter: "brightness(0.78) saturate(0.85)" }}
      onLoadedData={(e) => (e.currentTarget.style.opacity = "1")}
      onEmptied={(e) => (e.currentTarget.style.opacity = "0")}
    />
  );
}

/** The room: its camera video with the skeleton the model sees drawn over it. */
export function LiveSkeleton({ room, tone = "calm", className = "" }: { room: string; tone?: Tone; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const vid = useRef<HTMLVideoElement>(null);
  useAnimationFrame(() => {
    const c = ref.current;
    if (!c) return;
    const r = live[room];
    const f = r?.frame;
    const v = vid.current;
    if (v) {
      // frames stop when the camera is cut: freeze the video rather than loop
      const fresh = f && f.ct !== undefined && serverNow() - f.t < 0.5;
      syncVideo(v, f?.video, fresh ? f.ct! + Math.max(0, serverNow() - f.t) : null);
    }
    const { ctx, w, h } = sizeCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const onVideo = showing(v);
    const box = fit(w, h, frameAspect(v, useStore.getState().status[room]?.aspect ?? 16 / 9));
    if (!onVideo) {
      // the edge of the camera's view, as a hairline
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.lineWidth = 1;
      ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
    }
    if (!r) return;
    const lw = lineWidth(box.h);
    if (f?.present) {
      drawOver(ctx, f.kp, box, TONE[tone], 1, lw, onVideo);
    } else if (r.lastPresent) {
      // last seen: a fading ghost where the person was
      const gone = serverNow() - r.lastPresent.t;
      const a = Math.max(0.12, 0.45 - gone / 40);
      ctx.setLineDash([3, 4]);
      drawSkeleton(ctx, r.lastPresent.kp, box, TONE.muted, a, lw);
      ctx.setLineDash([]);
    }
    if (!f?.present) {
      ctx.font = "11px 'Geist Mono Variable', monospace";
      ctx.textAlign = "center";
      ctx.fillStyle = TONE.muted;
      ctx.fillText("OUT OF VIEW", w / 2, h / 2);
    }
  }, [room, tone]);
  const pos = className.includes("absolute") ? "" : "relative h-full w-full";
  return (
    <div className={`${pos} ${className}`}>
      <RoomVideo videoRef={vid} />
      <canvas ref={ref} className="absolute inset-0 block h-full w-full" />
    </div>
  );
}

type ReplayFrame = { t: number; kp: Keypoint[]; present: boolean; ct?: number; video?: string | null };

/** Loops the seconds before an alert, over the video of that moment when there is one. */
export function ReplaySkeleton({ frames, aspect, fps, className = "" }: { frames: ReplayFrame[]; aspect: number; fps: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const vid = useRef<HTMLVideoElement>(null);
  useAnimationFrame(() => {
    const c = ref.current;
    if (!c || !frames.length) return;
    const loopS = frames.length / fps + 1.2; // a short pause before it repeats
    const i = Math.floor(((performance.now() / 1000) % loopS) * fps);
    const f = frames[Math.min(i, frames.length - 1)];
    const v = vid.current;
    if (v) syncVideo(v, f.video, f.ct ?? null);
    const { ctx, w, h } = sizeCanvas(c);
    ctx.clearRect(0, 0, w, h);
    const onVideo = showing(v);
    const box = fit(w, h, frameAspect(v, aspect));
    if (!onVideo) {
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1);
    }
    if (f.present) drawOver(ctx, f.kp, box, TONE.calm, 0.9, lineWidth(box.h), onVideo);
    // progress hairline along the bottom
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.fillRect(box.x, box.y + box.h - 2, (box.w * Math.min(i, frames.length)) / frames.length, 2);
  }, [frames, aspect, fps]);
  return (
    <div className={`relative ${className}`}>
      <RoomVideo videoRef={vid} />
      <canvas ref={ref} className="absolute inset-0 block h-full w-full" />
    </div>
  );
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
