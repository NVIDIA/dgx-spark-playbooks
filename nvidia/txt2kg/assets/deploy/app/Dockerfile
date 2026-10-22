# Optimized multi-stage Dockerfile for faster builds
# Stage 1: Dependencies
FROM node:20-slim AS deps
WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@10 --activate

# Copy dependency files
COPY ./frontend/package.json ./frontend/pnpm-lock.yaml* ./

# Install dependencies with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm config set auto-install-peers true && \
    if [ -f pnpm-lock.yaml ]; then \
        pnpm install --no-optional --frozen-lockfile || pnpm install --no-optional --no-frozen-lockfile; \
    else \
        pnpm install --no-optional --no-frozen-lockfile; \
    fi

# Stage 2: Builder
FROM node:20-slim AS builder
WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@10 --activate

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/package.json ./package.json

# Copy source code
COPY ./frontend/ ./

# Set build environment variables
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

# Build with cache mount for Next.js cache
RUN --mount=type=cache,target=/app/.next/cache \
    pnpm build

# Stage 3: Production runtime
FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create non-root user for security
RUN groupadd --gid 1001 nodejs && \
    useradd --uid 1001 --gid nodejs --no-create-home nextjs

# Copy necessary files from builder
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Create data directory for query logs with proper permissions
RUN mkdir -p /app/data && chown -R nextjs:nodejs /app/data

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
