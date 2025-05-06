FROM alpine:latest
LABEL org.opencontainers.image.source="https://github.com/ynyeh0221/Vecraft"
LABEL org.opencontainers.image.description="test Docker image"
LABEL org.opencontainers.image.licenses="MIT"

CMD ["echo", "Hello GitHub Container Registry!"]
