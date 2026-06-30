# Week 7 Results

The completed adaptive run is preserved at
`adaptive_constrained_unlearning_v1/`. The rollback follow-up writes to the
separate `rollback_constrained_unlearning_v2/` folder.

Do not create the run folder by hand. The runner owns its rolling checkpoints,
candidate adapters, selection predictions, final evaluations, metrics, and
generated report.

The two run names, resume assets, notebooks, metrics, and reports are distinct.
Running v2 does not modify the v1 folder.
