import { MongoClient } from "mongodb";

export type TimelineItem = {
  id: string;
  title: string;
  detail?: string;
  timestamp: Date;
  kind: "event" | "summary";
};

type TimelineDocument = {
  _id: string;
  events?: {
    caption?: string;
    timestamp?: Date | string;
  }[];
};

type ProfileDocument = {
  _id?: unknown;
  name?: string;
  conversations?: {
    summary?: string;
    timestamp?: Date | string;
  }[];
};

type FetchTimelineOptions = {
  profileNameRegex?: RegExp;
};

const uri = process.env.MONGO_URI ?? "mongodb://localhost:27017";
const dbName = process.env.MONGO_DB ?? "tbd";

const capitalize = (value: string): string => {
  if (!value) {
    return value;
  }
  const lower = value.toLowerCase();
  return lower.charAt(0).toUpperCase() + lower.slice(1);
};

const formatProfileName = (name?: string): string | undefined => {
  if (!name) {
    return undefined;
  }
  const sanitized = name.includes("-") ? name.split("-")[0] ?? name : name;
  return capitalize(sanitized.trim());
};

declare global {
  // eslint-disable-next-line no-var
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

const clientPromise =
  globalThis._mongoClientPromise ??
  (globalThis._mongoClientPromise = new MongoClient(uri, {
    maxPoolSize: 5,
  }).connect());

export async function fetchTimeline(
  options: FetchTimelineOptions = {},
): Promise<TimelineItem[]> {
  const client = await clientPromise;
  const db = client.db(dbName);

  const timelines = db.collection<TimelineDocument>("timelines");
  const profilesCollection = db.collection<ProfileDocument>("profiles");

  const profileQuery: Record<string, unknown> = options.profileNameRegex
    ? { name: { $regex: options.profileNameRegex } }
    : {};

  const [timelineDoc, profiles] = await Promise.all([
    timelines.findOne({ _id: "timeline" }, { projection: { events: 1 } }),
    profilesCollection
      .find(profileQuery, { projection: { name: 1, conversations: 1 } })
      .toArray(),
  ]);

  const eventItems =
    timelineDoc?.events?.map((event, index) => ({
      id: `event-${index}`,
      title: event?.caption ?? "Untitled event",
      timestamp: event?.timestamp ? new Date(event.timestamp) : new Date(0),
      kind: "event" as const,
    })) ?? [];

  const summaryItems =
    profiles.flatMap((profile) => {
      const profileId = profile._id ? String(profile._id) : "profile";
      return (profile.conversations ?? []).map(
        (conversation, conversationIndex) => {
          const displayName = formatProfileName(profile.name);
          return {
            id: `${profileId}-${conversationIndex}`,
            title: conversation?.summary ?? "Summary",
            detail: displayName ? `Talked to ${displayName}` : undefined,
            timestamp: conversation?.timestamp
              ? new Date(conversation.timestamp)
              : new Date(0),
            kind: "summary" as const,
          };
        },
      );
    }) ?? [];

  return [...eventItems, ...summaryItems].sort(
    (a, b) => a.timestamp.getTime() - b.timestamp.getTime(),
  );
}

export type TimelineDay = {
  key: string;
  date: Date;
  label: string;
  items: TimelineItem[];
};

const dayLabelFormatter = new Intl.DateTimeFormat("en-US", {
  weekday: "long",
  month: "long",
  day: "numeric",
});

export function groupTimelineByDay(items: TimelineItem[]): TimelineDay[] {
  const byDay = new Map<string, TimelineItem[]>();

  items.forEach((item) => {
    // Use local date for grouping
    const year = item.timestamp.getFullYear();
    const month = String(item.timestamp.getMonth() + 1).padStart(2, '0');
    const day = String(item.timestamp.getDate()).padStart(2, '0');
    const dayKey = `${year}-${month}-${day}`;
    
    const existing = byDay.get(dayKey);
    if (existing) {
      existing.push(item);
    } else {
      byDay.set(dayKey, [item]);
    }
  });

  return Array.from(byDay.entries())
    .map(([key, dayItems]) => {
      const representative = dayItems[0]?.timestamp ?? new Date();
      const date = new Date(representative);
      return {
        key,
        date,
        label: dayLabelFormatter.format(date),
        items: dayItems.sort(
          (a, b) => a.timestamp.getTime() - b.timestamp.getTime(),
        ),
      };
    })
    .sort((a, b) => a.date.getTime() - b.date.getTime());
}

