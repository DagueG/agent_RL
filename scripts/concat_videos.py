import glob
import os

from moviepy import VideoFileClip, concatenate_videoclips

# Récupère tous les .mp4 du dossier videos
video_files = sorted(glob.glob("videos/eagle1_landing*.mp4"))
print(f"Vidéos trouvées : {len(video_files)}")
for v in video_files:
    print(f"  - {v}")

if not video_files:
    raise SystemExit("Aucune vidéo trouvée dans videos/")

# Charge et concatène
clips = [VideoFileClip(v) for v in video_files]
total_duration = sum(c.duration for c in clips)
print(f"\nDurée totale brute : {total_duration:.1f}s")

final_clip = concatenate_videoclips(clips, method="compose")

# Cible : 20-30 secondes. Si plus long, on coupe.
target_duration = min(final_clip.duration, 30)
final_clip = final_clip.subclipped(0, target_duration)

output_path = "videos/eagle1_demo.mp4"
final_clip.write_videofile(output_path, codec="libx264", fps=50)

for c in clips:
    c.close()
final_clip.close()

print(f"\nVidéo finale : {output_path} ({target_duration:.1f}s)")