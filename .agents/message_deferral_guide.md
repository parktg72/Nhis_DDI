# Sequential external-worker dispatch

Codex LO may have only one outbound Claude Code or AGY request in flight.

## Rules

1. Send a self-contained brief to one worker.
2. Until that worker reports completion or idle, queue all follow-up and other-worker messages locally.
3. After completion, Codex LO validates the result before sending the next request.
4. Workers never communicate with each other or the user and never spawn another worker.
5. Do not poll consuming bridge queues. Use the adapter process completion and result envelope.

Every worker result must include exact files changed, commands/tests run, validation status, risks, and one recommended next step.
