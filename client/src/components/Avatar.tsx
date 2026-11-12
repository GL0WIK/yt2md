import React from "react";
import { ListVideo } from "lucide-react";

/** Channel picture, or its initial when YouTube gives none or the image does not load. */
export function Avatar(props: { name: string; url?: string | null; playlist?: boolean; large?: boolean }) {
  const [broken, setBroken] = React.useState(false);
  const size = props.large ? "avatar large" : "avatar";
  if (props.url && !broken)
    return (
      <img
        className={size}
        src={props.url}
        alt=""
        referrerPolicy="no-referrer"
        onError={() => setBroken(true)}
      />
    );
  return (
    <span className={size + " avatar-letter"} aria-hidden="true">
      {props.playlist ? (
        <ListVideo size={props.large ? 34 : 22} />
      ) : (
        (props.name.trim()[0] ?? "?").toUpperCase()
      )}
    </span>
  );
}
