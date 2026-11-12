export type {
  AnalysisProgress,
  AnalysisRun,
  AnalyzeVideoResult,
  AnalyzeVideoSuccess,
  CatalogVideo,
  CatalogPage,
} from "../../server/src/shared/schemas";
import type { AnalysisRun } from "../../server/src/shared/schemas";
export type RunSummary = AnalysisRun & { successCount: number; errorCount: number };
