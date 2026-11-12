"""Real FFmpeg smoke check with a generated clip; no network or model weights."""
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
import media
from common import file_sha256

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    raise SystemExit("Installer FFmpeg et FFprobe avant ce test.")
temporary_root = ROOT / ".smoke-tmp"
temporary_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=temporary_root) as temp:
    directory = Path(temp)
    config = {"ffmpegBinary": "ffmpeg", "ffprobeBinary": "ffprobe", "downloadTimeoutMs": 120000,
              "frameMaxSide": 640, "fps": 1}
    for duration in (0.2, 7, 7.6):
        source = directory / (str(duration) + ".mkv")
        media.run("ffmpeg", ["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(duration),
                            "-c:v", "libx264", "-c:a", "pcm_s16le", str(source)], 60)
        result = media.extract(source, directory / (str(duration) + "-extracted"), config, file_sha256(source))
        assert len(result["frames"]) == result["durationSeconds"]
        assert result["frames"][0]["startMs"] == 0
        assert result["frames"][-1]["endMs"] == result["durationMs"]
        assert all(frame["ptsMs"] == frame["index"] * 1000 for frame in result["frames"])
        chunk = directory / (str(duration) + ".wav")
        media.slice_wave(directory / (str(duration) + "-extracted/audio.wav"), chunk, 0, result["durationMs"])
        print(f"FFmpeg OK : {duration}s, {len(result['frames'])} captures, PCM mono 16 kHz.")
