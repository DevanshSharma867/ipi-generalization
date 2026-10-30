# Draft (not sent): email to Zhuowen Liu about the AgentDojo pairing

**Subject:** benign-instruction-bench: injection-task pairing in collect_agentdojo_pairs.py

Dear Dr. Liu,

Thank you for releasing benign-instruction-bench, with the code, the score files and the pinned environment. We are
re-running it as the transfer benchmark for our own study of prompt-injection detectors, and it has been easy to
reproduce: in your environment, with your scripts, PIGuard, Prompt Guard 2 86M and Horizon-Labs match your released
scores to about 1e-5 per row on BIPIA and TB-Matched.

AD-Matched is the one set we could not match row for row. `scripts/collect_agentdojo_pairs.py` assigns each user task
an injection task with `inj_tasks[hash(task_id) % len(inj_tasks)]`. Python randomizes `hash()` of strings per process
unless `PYTHONHASHSEED` is set. We rebuilt with `PYTHONHASHSEED=0` and Python 3.12, as `reproduce.sh` and the README
say. Row counts and labels match your release exactly, but most injected rows score differently (for example, 273 of
432 for PIGuard), while the benign rows match. We think the released score files may come from a run without the
fixed seed, so the pairing differs. AD-Shift is built the same way and is probably affected too.

Could you share the pairing (user task to injection task, per suite) or the seed behind the released AD-Matched and
AD-Shift scores? If you have the built `agentdojo_pairs.jsonl` and `agentdojo_pairs_shift.jsonl`, those would settle
it as well. We are also trying to recover the pairing from your released scores and will send you the result,
whichever way it goes.

Thank you again for making the work so reproducible.

Best regards,
Devansh Sharma
BITS Pilani, Hyderabad Campus
