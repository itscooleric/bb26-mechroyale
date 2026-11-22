# ericiscool (Mech Royale bot)

A bot for the Builders Buildathon Mech Royale track (protocol `mechfight/1`).

- `bot.py`: the bot.
- `world.py`: map loading, line of sight and a walk grid built from `/arena`.
- `mechfight.py`: the official bot-kit wrapper, copied unchanged from the event's `mechfight-kit`.

Python standard library only. Build and run:

```sh
docker build -t ericiscool .
docker run --rm -i --network none -v /path/to/arena:/arena:ro ericiscool
```

MIT licensed (see `LICENSE`).
