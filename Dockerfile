# Mech Royale (mechfight/1) bot. Standard library only: nothing to install.
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, uid 65534, and the map at /arena.
FROM python:3.13-slim
WORKDIR /bot
COPY mechfight.py world.py bot.py ./
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
