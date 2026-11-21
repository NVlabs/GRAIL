# Motion data export

For an existing rollout, `grail.data_export.export_successful_rollouts` extracts robot and object motion. To merge locally exported shards, run `python -m grail.data_export.merge_exports --help` for the required experiment and source directories. Inspect the merged manifest and overlap counts before using a library.

For native retargeting output, use the [manipulation handoff](retargeting.md#hand-off-a-retargeted-motion). It preserves the source clock and object geometry needed by a downstream tracker.
