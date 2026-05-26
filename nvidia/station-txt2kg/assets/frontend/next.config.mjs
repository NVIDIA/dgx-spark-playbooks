/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Configure external packages for server components
  serverExternalPackages: ['@langchain/community'],
  experimental: {
    // webpackBuildWorker: true,
  },
  // Enable standalone output for optimized Docker builds
  output: 'standalone',
  // Make environment variables accessible to server components
  env: {
    NVIDIA_API_KEY: process.env.NVIDIA_API_KEY,
  },
  // Remove API route timeout limits for large model processing
  serverRuntimeConfig: {
    // No duration limit - let large models complete naturally
    maxDuration: 0,
  },
}

// Define environment variables that should be available to the client
const clientEnv = {
  NVIDIA_API_KEY: process.env.NVIDIA_API_KEY,
  // Other environment variables as needed
};

export default nextConfig
