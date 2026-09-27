# Recursive owner runtime attempts

- [1] 27B base-model smoke with `llama-glm5 -no-cnv` -> CLI rejected the argument before model load; no model/runtime conclusion.
- [2] Same bounded smoke using `--no-conversation` -> this fork also rejects the flag before model load; change strategy instead of retrying completion toggles.
- [3] Same bounded smoke in default completion mode using `llama-glm5/build-vulkan` -> model hash verified, but fork rejected custom GGUF tensor type 142 on `output.weight`; this runtime/format pair is invalid. Change strategy: identify the PQ2-capable fork/build from source and E3 provenance instead of trying binaries.
