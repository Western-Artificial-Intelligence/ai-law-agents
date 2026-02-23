# 3MT Script (Full Draft) - B.A.I.L.I.F.F.

## Target length
- ~400-440 words for a 3:00 delivery.

## Full 3-minute script
Imagine two defendants in the same courtroom.  
Same facts. Same witnesses. Same law.  
Now change only one detail: a demographic cue in a name or dialect.

Should the verdict change?  
No.

Should the quality of the trial process change?  
Also no.

That is the core problem my team studied with **B.A.I.L.I.F.F.**:  
Bias Analysis in Interactive Legal Intelligence and Fairness Framework.

Most AI fairness checks are static.  
They ask one prompt, get one answer, and score the output.  
But legal decision-making is not a one-turn quiz.  
It is interactive, adversarial, and procedural.

So we built a mock AI courtroom with three agents: Judge, Prosecution, and Defense.  
For each case, we run paired trials where the facts are held constant and only demographic cues are toggled.

That design lets us isolate whether cues alone can shift behavior.

We measure two things.

First, **outcome direction**: does conviction probability move, and by how much?  
Second, **process stability**: how often does the verdict flip under cue swaps, even when facts are identical?

Our most important result is not one dramatic biased model.  
It is a structural pattern:

Across model families, pooled directional effects are generally modest,  
but flip rates are consistently non-zero.

In our panel, family-level flip rates range roughly from **0.149 to 0.383**.  
So even when average directional shifts look small, instability remains.

That means a system can look acceptable in aggregate metrics while still being unreliable at the individual-case level.

And in legal settings, that is not a cosmetic issue.  
If two near-identical defendants can get different outcomes because of cue sensitivity, that is a procedural fairness failure.

So the takeaway is direct:

Fair legal AI requires **interactive adversarial auditing**, not only static fairness scoring.  
We must evaluate both **what** decision is made and **how** that decision is reached under controlled counterfactual swaps.

When we ask, "Can AI hold a fair trial?"  
Our answer is: not unless we audit it like a courtroom, not like a benchmark quiz.

## Timing map (recommended)
- `0:00-0:35` Hook + fairness principle
- `0:35-1:20` What B.A.I.L.I.F.F. is and why static audits fail
- `1:20-2:20` Method + main empirical result (flip-rate stability)
- `2:20-3:00` Why it matters + final takeaway

## Delivery notes
- Point to the chart at `1:30` and say: "These bars are instability under controlled swaps."
- Pause after "non-zero" and after "procedural fairness failure."
- Keep the close sentence slow and clean.
