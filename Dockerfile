FROM alpine:latest
LABEL org.opencontainers.image.source="https://github.com/ynyeh0221/vecraft"
LABEL org.opencontainers.image.description="test Docker image"
LABEL org.opencontainers.image.licenses="MIT"

CMD ["echo", "Hello GitHub Container Registry!"]
