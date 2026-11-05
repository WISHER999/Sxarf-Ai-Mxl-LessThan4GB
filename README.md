# Sxarf AI

Setup: `./setup.sh` · Run: `npm start` · Tests: `npm test`

## Windows build

The Windows x64 build uses a separate CPU backend based on llama.cpp; MLX
features remain on the Apple Silicon macOS backend. On first launch the app
downloads a verified private Python runtime, installs its dependencies, and
downloads the Llama 3.2 3B GGUF model. An internet connection is required.

To build the installer from source, install Node.js and run:

```powershell
npm install
npm run build:win
```

The NSIS installer is written to `dist/`. Windows currently supports local
text chat and CoCode; web research agents, image analysis, and local speech
transcription are not included. Inference is CPU-only.

The MLX server keeps one model resident at a time. Research specialists load
sequentially and are released before the selected chat model is restored. The
default chat model is Llama 3.2 3B; model choices are in the app's model menu.

MLX allocation defaults are 2.5 GiB active memory and 0.25 GiB cache. Both can
be lowered with `MLX_MEMORY_LIMIT_GB` and `MLX_CACHE_LIMIT_GB` (upper-capped by
the server). Context, generation, and KV-cache settings are similarly bounded
in `server/server.py`. These limits are not a substitute for measuring total
process-tree RSS on the target Mac.

With the server running, `npm run bench` measures short chat, an approximately
2k-token prompt, and research chat. It reports TTFT, generation throughput,
process-tree peak RSS, and MLX peak allocation. For a manually started server,
run `python server/bench.py --pid SERVER_PID`.

Research controls: **Agents: Auto / Always / Off**. Auto routes likely
research questions to local specialists; web search uses keyless search-engine
HTML endpoints and can be unavailable. Inspect `/agents`, `/agents/status`,
and `/agents/trace` on the local server.
