# MomentWire Renderer

Disposable GitHub Actions renderer for MomentWire audience-growth clips.

The VPS remains the source of truth for scouting, rights, queue state, Drive archive and publishing. GitHub Actions performs creative rendering only on ephemeral hosted runners.

## Contract

- Input: one rights-cleared `momentwire-streamer-creative-request/v1` batch.
- Render at least two materially distinct 1080x1920 variants.
- Preserve source content and original audio.
- Never crop-to-fill when it would cut faces or source UI/text.
- Captions and hooks must remain inside platform-safe zones.
- Reject bad crops, visual collisions, malformed captions and technical failures.
- Return exact SHA-256, byte size, duration and QA evidence.
- No platform credentials or private source files are committed to this repository.

The repository is designed to be public so standard GitHub-hosted runner usage can stay free under GitHub's public-repository Actions policy.
