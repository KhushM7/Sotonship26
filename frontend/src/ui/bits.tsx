import { useEffect, useState, type ReactNode } from "react";
import type { Incident } from "../contracts";
import { clock, firstName, timeOfDay } from "../format";
import { serverNow } from "../store";

/** Server time, re-rendered every half second. */
export function useNow(ms = 500) {
  const [now, setNow] = useState(serverNow);
  useEffect(() => {
    const id = setInterval(() => setNow(serverNow()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}

export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`font-mono text-[11px] uppercase tracking-[0.08em] text-ink-400 ${className}`}>
      {children}
    </div>
  );
}

/** Colour, word and shape together: never colour alone. */
export function LevelMark({ level, quiet = false }: { level: string; quiet?: boolean }) {
  const tone =
    level === "CHECK" ? "text-caution" : level === "OFFLINE" ? "text-ink-300" : "text-alarm";
  return (
    <span className={`inline-flex items-center gap-1.5 font-mono text-[11px] font-medium uppercase tracking-[0.08em] ${quiet ? "text-ink-300" : tone}`}>
      <span
        className={`inline-block size-1.5 ${level === "CHECK" ? "rotate-45" : level === "OFFLINE" ? "" : "rounded-full"} bg-current`}
      />
      {level}
    </span>
  );
}

const STEPS = [
  { key: "alert", label: "Alert" },
  { key: "going", label: "On the way" },
  { key: "with", label: "With them" },
  { key: "done", label: "Resolved" },
] as const;

/** The incident's journey as four steps with who and when. */
export function Lifecycle({ inc, compact = false }: { inc: Incident; compact?: boolean }) {
  const at: (number | null)[] = [inc.opened_at, inc.acknowledged_at, inc.attending_at, inc.resolved_at];
  const who = [null, inc.acknowledged_by, inc.attending_by, inc.resolved_by];
  const current = at.findLastIndex((t) => t !== null);
  return (
    <ol className="grid grid-cols-4 gap-2">
      {STEPS.map((s, i) => {
        const t = at[i];
        const done = t !== null;
        return (
          <li key={s.key} className="min-w-0">
            <div className={`h-0.5 ${done ? (i === current && inc.state !== "resolved" ? "bg-ink-100" : "bg-ink-400") : "bg-ink-700"}`} />
            <div className={`mt-1.5 truncate text-[11px] ${done ? "text-ink-200" : "text-ink-500"}`}>{s.label}</div>
            {!compact && (
              <div className="truncate font-mono text-[10px] text-ink-400">
                {t === null ? " " : i === 0 ? timeOfDay(t) : `+${clock(t - inc.opened_at)} ${firstName(who[i])}`}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function Button({
  children,
  onClick,
  kind = "primary",
  size = "md",
  disabled,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  kind?: "primary" | "alarm" | "ghost";
  size?: "md" | "lg";
  disabled?: boolean;
  className?: string;
}) {
  const kinds = {
    primary: "bg-ink-100 text-ink-950 hover:bg-white",
    alarm: "bg-alarm text-ink-950 hover:brightness-110",
    ghost: "border border-ink-700 text-ink-200 hover:border-ink-500 hover:text-ink-100",
  };
  const sizes = { md: "h-10 px-4 text-sm", lg: "h-14 px-5 text-base" };
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-md font-medium transition-[transform,background-color,border-color] duration-150 active:scale-[0.98] disabled:opacity-40 ${kinds[kind]} ${sizes[size]} ${className}`}
    >
      {children}
    </button>
  );
}
