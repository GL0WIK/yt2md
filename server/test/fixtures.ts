import { CatalogVideo, VideoAnalysis } from "../src/shared/schemas.js";

export const fixtureVideo: CatalogVideo = {
  id: "abc123def45",
  title: "Video de test",
  url: "https://www.youtube.com/watch?v=abc123def45",
  channel: "Chaine test",
  publishedAt: "2026-05-01",
  durationSeconds: 120,
  description: "Description publique",
  publicStats: {
    views: 1000,
    likes: null,
    comments: 12,
  },
  source: "direct",
};

export const fixtureAnalysis: VideoAnalysis = {
  metadata: {
    title: "Video de test",
    url: "https://www.youtube.com/watch?v=abc123def45",
    channel: "Chaine test",
    publicationDate: "2026-05-01",
    durationSeconds: 120,
  },
  publicStats: {
    views: 1000,
    likes: null,
    comments: 12,
  },
  youtubeDescription: "Description publique",
  analysisSettings: {
    model: "Qwen/Qwen3.5-9B",
    fps: 0.333,
    mediaResolution: "video_1fps_7frames",
  },
  transcript: [
    {
      speaker: "Narrateur",
      timestamp: "00:01",
      text: "Bonjour",
    },
    {
      speaker: "Narrateur",
      timestamp: "01:50",
      text: "Au revoir",
    },
  ],
  visualSegments: [
    {
      timestamp: "00:00",
      endTimestamp: "00:15",
      title: "Introduction",
      visualDescription: "Plan d'ouverture",
      screenText: ["Titre"],
    },
    {
      timestamp: "00:15",
      endTimestamp: "00:30",
      title: "Demonstration",
      visualDescription: "Plan de demonstration",
      screenText: [],
    },
    {
      timestamp: "00:30",
      endTimestamp: "00:45",
      title: "Detail",
      visualDescription: "Plan de detail",
      screenText: [],
    },
    {
      timestamp: "00:45",
      endTimestamp: "01:00",
      title: "Interface",
      visualDescription: "Interface a l'ecran",
      screenText: [],
    },
    {
      timestamp: "01:00",
      endTimestamp: "01:15",
      title: "Manipulation",
      visualDescription: "Manipulation visible",
      screenText: [],
    },
    {
      timestamp: "01:15",
      endTimestamp: "01:30",
      title: "Explication",
      visualDescription: "Plan d'explication",
      screenText: [],
    },
    {
      timestamp: "01:30",
      endTimestamp: "01:45",
      title: "Transition",
      visualDescription: "Transition visuelle",
      screenText: [],
    },
    {
      timestamp: "01:45",
      endTimestamp: "02:00",
      title: "Capture finale",
      visualDescription: "Capture d'ecran",
      screenText: [],
    },
  ],
};
