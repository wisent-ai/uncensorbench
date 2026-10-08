FROM rust:slim AS build

# The revision the page is built from. A Space passes its variables to the
# build as arguments, so the Space's settings state it; nothing is built from
# a branch whose contents can change under the same name. An unset variable
# fails this step, which names it.
ARG UNCENSORBENCH_REVISION
RUN test -n "$UNCENSORBENCH_REVISION"
RUN apt-get update && apt-get install --yes --no-install-recommends git ca-certificates pkg-config \
 && cargo install --locked --git https://github.com/wisent-ai/uncensorbench --rev "$UNCENSORBENCH_REVISION" uncensorbench

FROM debian:stable-slim
COPY --from=build /usr/local/cargo/bin/uncensorbench /usr/local/bin/uncensorbench
WORKDIR /app
COPY leaderboard.csv .

# The address the page listens on is the Space's: Hugging Face routes a
# Docker Space to the port its README's app_port names, or to the default it
# states (https://huggingface.co/docs/hub/spaces-sdks-docker), so the Space
# variable LEADERBOARD_LISTEN states host and port to match. Unset, the build
# stops here.
ARG LEADERBOARD_LISTEN
RUN test -n "$LEADERBOARD_LISTEN"
ENV LEADERBOARD_LISTEN=$LEADERBOARD_LISTEN

CMD uncensorbench leaderboard serve --csv leaderboard.csv --listen "$LEADERBOARD_LISTEN"
