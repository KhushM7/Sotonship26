import type { Keypoint } from "./contracts";

export const BONES: [number, number][] = [
  [1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8],
  [7, 8], [7, 9], [9, 11], [8, 10], [10, 12],
];

/**
 * The area of a w x h canvas that shows a source frame of the given aspect
 * (width / height), letterboxed so the skeleton is never stretched.
 */
export function fit(w: number, h: number, aspect: number) {
  let fw = w;
  let fh = w / aspect;
  if (fh > h) {
    fh = h;
    fw = h * aspect;
  }
  return { x: (w - fw) / 2, y: (h - fh) / 2, w: fw, h: fh };
}

/** Draw one 13-point skeleton (normalised coordinates) into the box. */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  kp: Keypoint[],
  box: { x: number; y: number; w: number; h: number },
  color: string,
  alpha = 1,
  width = 2,
) {
  const pt = (k: NonNullable<Keypoint>) => [box.x + k[0] * box.w, box.y + k[1] * box.h] as const;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = width;
  for (const [a, b] of BONES) {
    const ka = kp[a];
    const kb = kp[b];
    if (!ka || !kb) continue;
    // low-visibility joints are drawn faint rather than hidden
    ctx.globalAlpha = alpha * Math.max(0.25, Math.min(ka[2], kb[2]));
    const [x1, y1] = pt(ka);
    const [x2, y2] = pt(kb);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  const nose = kp[0];
  const ls = kp[1];
  const rs = kp[2];
  if (nose) {
    ctx.globalAlpha = alpha * Math.max(0.3, nose[2]);
    const [nx, ny] = pt(nose);
    const shoulder = ls && rs ? Math.hypot((ls[0] - rs[0]) * box.w, (ls[1] - rs[1]) * box.h) : 20;
    const r = Math.max(3, Math.min(shoulder * 0.32, box.h * 0.05));
    ctx.beginPath();
    ctx.arc(nx, ny, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

/** Size a canvas's backing store to its CSS size times the device pixel ratio. */
export function sizeCanvas(canvas: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
