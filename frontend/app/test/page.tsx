import Link from "next/link";
import { MeshGradient } from "@paper-design/shaders-react";

import { fetchTimeline, groupTimelineByDay } from "../../lib/timeline";

const dateFormatter = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
});

type PageProps = {
  searchParams?:
    | Record<string, string | string[] | undefined>
    | Promise<Record<string, string | string[] | undefined>>;
};

export default async function TestTimeline({ searchParams }: PageProps) {
  const timeline = await fetchTimeline({ profileNameRegex: /test/i });
  const timelineByDay = groupTimelineByDay(timeline);
  const resolvedParams =
    searchParams && typeof (searchParams as Promise<unknown>).then === "function"
      ? await (searchParams as Promise<Record<string, string | string[] | undefined>>)
      : (searchParams as Record<string, string | string[] | undefined>) ?? {};
  const rawDay = resolvedParams.day;
  const dayKey = Array.isArray(rawDay) ? rawDay[0] ?? undefined : rawDay ?? undefined;

  let dayIndex = 0;
  if (dayKey) {
    const foundIndex = timelineByDay.findIndex((day) => day.key === dayKey);
    dayIndex = foundIndex >= 0 ? foundIndex : 0;
  }
  const currentDay = timelineByDay[dayIndex];
  const previousDay = dayIndex > 0 ? timelineByDay[dayIndex - 1] : null;
  const nextDay =
    dayIndex < timelineByDay.length - 1 ? timelineByDay[dayIndex + 1] : null;

  return (
    <div className="relative min-h-screen py-16 text-zinc-900">
      <div className="fixed inset-0 -z-10">
        <MeshGradient
          colors={["#f0f4ff", "#f2f0ff", "#ffffff", "#ffffff"]}
          distortion={0.23}
          swirl={0.1}
          grainMixer={0}
          grainOverlay={0}
          speed={0.24}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
      <main className="relative mx-auto flex w-full max-w-2xl flex-col gap-12 px-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">
            {currentDay ? currentDay.label : "Timeline"}
          </h1>
          <p className="text-md text-zinc-500 mt-2 mb-8">
            Here is what you did today.
          </p>
          {timelineByDay.length > 1 ? (
            <nav className="flex items-center justify-between text-xs uppercase tracking-wide text-zinc-400">
              <span className="flex items-center gap-2">
                <span aria-hidden="true">←</span>
                {previousDay ? (
                  <Link
                    href={`?day=${previousDay.key}`}
                    className="hover:text-zinc-700"
                  >
                    {previousDay.label}
                  </Link>
                ) : (
                  <span className="opacity-50" aria-disabled="true">
                    No earlier entries
                  </span>
                )}
              </span>
              <span>
                Day {dayIndex + 1} of {timelineByDay.length}
              </span>
              <span className="flex items-center gap-2">
                {nextDay ? (
                  <Link
                    href={`?day=${nextDay.key}`}
                    className="hover:text-zinc-700"
                  >
                    {nextDay.label}
                  </Link>
                ) : (
                  <span className="opacity-50" aria-disabled="true">
                    No later entries
                  </span>
                )}
                <span aria-hidden="true">→</span>
              </span>
            </nav>
          ) : null}
        </header>

        <section className="relative border-l border-zinc-200 pl-6">
          {!currentDay || currentDay.items.length === 0 ? (
            <p className="text-sm text-zinc-500">No timeline entries yet.</p>
          ) : (
            currentDay.items.map((item, index) => (
              <article
                key={item.id}
                className="relative pb-10 last:pb-0"
                style={{
                  marginTop: index === 0 ? 0 : "0.5rem",
                }}
              >
                <span className="absolute -left-3 top-2 h-2.5 w-2.5 rounded-full border border-white bg-zinc-900 shadow-sm" />
                <time className="text-xs uppercase tracking-wide text-zinc-400">
                  {dateFormatter.format(item.timestamp)}
                </time>
                <h2 className="mt-2 text-base font-medium leading-6 text-zinc-900">
                  {item.title}
                </h2>
                <div className="mt-1 text-sm text-zinc-500">
                  {item.detail ?? (item.kind === "event" ? "Observation" : "Summary")}
                </div>
              </article>
            ))
          )}
        </section>
      </main>
    </div>
  );
}

