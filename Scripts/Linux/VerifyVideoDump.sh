#!/bin/bash
# usage: verify-dump.sh DUMPDIR   (the directory passed to --dump)
# Per case: decodes stream.* with ffmpeg (cropped to the video size) and reports whether it is bit-exact to the NRI decode (decoded.yuv)
# and its luma PSNR against source.yuv. Needs ffmpeg and ffprobe.
D=$1
for c in "$D"/*/; do
  c=${c%/}; n=$(basename "$c"); s=$(ls "$c"/stream.* 2>/dev/null | head -1)
  [ -z "$s" ] || [ ! -f "$c/info.txt" ] && continue
  v=$(sed -n 's/^video=//p' "$c/info.txt"); w=${v%x*}; h=${v#*x}
  if [ ! -s "$c/decoded.yuv" ]; then echo "$n: encode-only, stream $(stat -c %s "$s") bytes"; continue; fi
  ffmpeg -v error -y -i "$s" -vf crop=$w:$h:0:0 -f rawvideo -pix_fmt nv12 "$c/ffmpeg.yuv" 2>"$c/ffmpeg.log"
  if cmp -s "$c/ffmpeg.yuv" "$c/decoded.yuv"; then ex="bit-exact"; else ex="DIFFERENT"; fi
  p=$(ffmpeg -hide_banner -f rawvideo -pix_fmt nv12 -s $v -i "$c/source.yuv" -f rawvideo -pix_fmt nv12 -s $v -i "$c/ffmpeg.yuv" -lavfi psnr -f null - 2>&1 | grep -o "y:[0-9.]*" | head -1)
  echo "$n: $v ffmpeg-vs-NRI-decode=$ex ffmpeg-psnr-$p"
done
