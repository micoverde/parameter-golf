# wdq_serious_eval2048

Serious-lane treatment on top of `WarmdownQuantization`.

The training recipe stays fixed. The only intended change is evaluation context:

- control: `eval@1408`, sliding stride `64`
- treatment: `eval@2048`, sliding stride `64`

The repo's own notes suggest undertrained single-GPU runs can benefit from the
longer eval context even when the fully trained 8xH100 recipe prefers `1408`.
