#!/usr/bin/env bash
set -e

# Ensure /config exists
mkdir -p /config

# 1) Copy default config files if they are missing in /config
if [ -d /app/config ]; then
  # Main docker config
  if [ -f /app/config/config.docker.yaml ] && [ ! -f /config/config.docker.yaml ]; then
    cp /app/config/config.docker.yaml /config/
  fi

  # Optionally also copy generic config.yaml if you want it on the host
  if [ -f /app/config/config.yaml ] && [ ! -f /config/config.yaml ]; then
    cp /app/config/config.yaml /config/
  fi

  # Stain JSONs
  for f in stain_list.json stain_error_replacements.json; do
    if [ -f "/app/config/$f" ] && [ ! -f "/config/$f" ]; then
      cp "/app/config/$f" "/config/$f"
    fi
  done
fi

# 2) Copy model files into /config/model (so the user sees them on the host, if mounted)
if [ -d /app/model ]; then
  mkdir -p /config/model
  for f in /app/model/*; do
    bn="$(basename "$f")"
    if [ ! -f "/config/model/$bn" ]; then
      cp "$f" "/config/model/$bn"
    fi
  done
fi

# 3) Copy ROI sets into /config/roiset
if [ -d /app/roiset ]; then
  mkdir -p /config/roiset
  for f in /app/roiset/*; do
    bn="$(basename "$f")"
    if [ ! -f "/config/roiset/$bn" ]; then
      cp "$f" "/config/roiset/$bn"
    fi
  done
fi

# 4) Finally, run the Babel-Shark CLI
exec python -m babel_shark.cli "$@"
