# assets

Small calibration / reference data goes here (reference trajectories for fusion tasks, tiny fixtures). Large or private HFM files are NOT committed.

For each file document: source, license, SHA-256, fields/units, and the task/shot it belongs to. Reference tracks are loaded with
`pgofp.envs.fusion.reference.ReferenceTrack`, which records a content hash into the run manifest. Nothing is provided yet (no HFM data has been supplied).
