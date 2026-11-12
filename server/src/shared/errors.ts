export class HttpError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "HttpError";
  }
}

// Bot check, rate limit, or a refused media URL once YouTube starts throttling this machine.
const YOUTUBE_BLOCKED =
  /Sign in to confirm you.{0,3}re not a bot|HTTP Error 429|Too Many Requests|unable to download video data: HTTP Error 403/i;
export const YOUTUBE_BLOCKED_MESSAGE =
  "YouTube bloque temporairement les lectures depuis cette machine (protection anti-robots). " +
  "Réessayez dans quelques heures, ou renseignez YT_DLP_COOKIES_FROM_BROWSER (ou YT_DLP_COOKIES) " +
  "dans .env puis redémarrez.";
export function isYoutubeBlocked(text: string): boolean {
  return YOUTUBE_BLOCKED.test(text);
}
