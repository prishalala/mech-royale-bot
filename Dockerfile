# Submission image: builds the bot in bot/. Pure standard-library Python, nothing to install.
# The match mounts the map read-only at /arena and runs us as uid 65534 on a read-only root.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /bot
COPY bot/mechfight.py bot/worldmap.py bot/bot.py ./
RUN python3 -m compileall -q /bot && chmod -R a+rX /bot
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
