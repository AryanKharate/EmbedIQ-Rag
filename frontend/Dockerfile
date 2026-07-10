# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps

# Copy all source files
COPY . .

# Build using node-server Nitro preset
ENV DOCKER_BUILD=1
RUN npm run build

# ── Stage 2: Run ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS runner

WORKDIR /app

# Copy only the compiled server output
COPY --from=builder /app/.output /app/.output

# Expose the Nitro server port
EXPOSE 3000

# The RAG backend URL — overrideable at runtime via docker-compose env
ENV NITRO_HOST=0.0.0.0
ENV NITRO_PORT=3000

CMD ["node", ".output/server/index.mjs"]
