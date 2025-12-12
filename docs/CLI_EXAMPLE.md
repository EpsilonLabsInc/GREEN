# Command Line Usage Example

This document shows a complete example of running GREEN from the command line with Azure OpenAI.

## Command

```bash
python run.py \
  --input-csv ~/Downloads/hand_data_filtering.csv \
  --gt-column answer_gt \
  --comparison-columns "answer_gt,base,clean_degen,clean_degen_excl_nofindings" \
  --output-csv green_results.csv \
  --verbose
```

## Full Output

```
Loading input CSV: /Users/ruian/Downloads/hand_data_filtering - Sheet3.csv
Initializing GREEN scorer...
[VERBOSE] Initializing GREEN scorer...
[VERBOSE] Batch size: 512
[VERBOSE] Parallel requests: True
[VERBOSE] use_azure=True, initializing Azure OpenAI client...
[VERBOSE] Loading Azure OpenAI configuration...
[VERBOSE] Azure config loaded: endpoint=https://epsilon-eastus-2.openai.azure.com/
[VERBOSE] Azure OpenAI client created successfully
[VERBOSE] Model: gpt-5.1, Deployment: gpt-5.1
Initialized Azure OpenAI client with model: gpt-5.1

============================================================
Analyzing column: answer_gt
============================================================
[VERBOSE] __call__ invoked with 485 references and 485 hypotheses
Processing data...making prompts
[VERBOSE] Dataset created with 485 examples
Map: 100%|████████████████████████████████████████| 485/485 [00:00<00:00, 19891.63 examples/s]
[VERBOSE] Prompts generated
Done.
[VERBOSE] Starting inference...
[VERBOSE] _infer_azure started
[VERBOSE] Processing 485 examples in 1 batches (batch_size=512)
==== Beginning Inference ====
0it [00:00, ?it/s][VERBOSE] Processing batch 1/1...
[VERBOSE] Sending 485 requests in parallel...
Parallel requests: 100%|██████████████████████████| 485/485 [00:08<00:00, 56.17it/s]
[VERBOSE] All 485 parallel requests completed
[VERBOSE] Batch 1 completed. Total completions: 485
1it [00:11, 11.39s/it]
==== End Inference ====
Computing summary ...
Seconds per example:  0.023675074528173072

[answer_gt] Mean GREEN score: 0.9938 ± 0.0784

-------------gpt-5.1----------------
 [Summary]: Green average 0.9938144329896907 and standard deviation 0.07840475604878987
 [Clinically Significant Errors Analyses]: <accuracy>. <representative error>

(a) False report of a finding in the candidate: 1.0.
 None

(b) Missing a finding present in the reference: 1.0.
 None

(c) Misidentification of a finding's anatomic location/position: 1.0.
 None

(d) Misassessment of the severity of a finding: 1.0.
 None

(e) Mentioning a comparison that isn't in the reference: 1.0.
 None

(f) Omitting a comparison detailing a change from a prior study: 1.0.
 None

----------------------------------


============================================================
Analyzing column: base
============================================================
[VERBOSE] __call__ invoked with 485 references and 485 hypotheses
Processing data...making prompts
[VERBOSE] Dataset created with 485 examples
Map: 100%|████████████████████████████████████████| 485/485 [00:00<00:00, 54220.31 examples/s]
[VERBOSE] Prompts generated
Done.
[VERBOSE] Starting inference...
[VERBOSE] _infer_azure started
[VERBOSE] Processing 485 examples in 1 batches (batch_size=512)
==== Beginning Inference ====
0it [00:00, ?it/s][VERBOSE] Processing batch 1/1...
[VERBOSE] Sending 485 requests in parallel...
Parallel requests: 100%|██████████████████████████| 485/485 [00:15<00:00, 31.62it/s]
[VERBOSE] All 485 parallel requests completed
[VERBOSE] Batch 1 completed. Total completions: 485
1it [00:15, 15.88s/it]
==== End Inference ====
Computing summary ...
Seconds per example:  0.0752577240934077

[base] Mean GREEN score: 0.7940 ± 0.2488

-------------gpt-5.1----------------
 [Summary]: Green average 0.7940203954121479 and standard deviation 0.24883830831382875
 [Clinically Significant Errors Analyses]: <accuracy>. <representative error>

(a) False report of a finding in the candidate: 0.6701030927835051.
  Stating "Remaining joint spaces are unremarkable" when the reference describes
  mild-to-moderate degenerative changes of multiple PIP and DIP joints and
  radiocarpal joint space narrowing.

(b) Missing a finding present in the reference: 0.6701030927835051.
  Omission of "Mild degenerative changes of multiple PIP and DIP joints"

(c) Misidentification of a finding's anatomic location/position: 0.8701030927835052.
  Degenerative changes are reported at the interphalangeal joint of the left thumb
  in the candidate, whereas the reference specifies mild degenerative changes at
  the first carpometacarpal joint.

(d) Misassessment of the severity of a finding: 0.8556701030927835.
  Candidate describes only "Mild degenerative joint space narrowing involving the
  third distal interphalangeal joint" and implies no other joint space narrowing,
  whereas the reference reports "Moderate" narrowing of the third DIP and "Mild"
  narrowing of additional PIP and DIP joints.

(e) Mentioning a comparison that isn't in the reference: 0.9979381443298969.
  The candidate report states "Fracture lucency is still visualized but less
  apparent, suggesting some interval healing," which implies interval improvement,
  whereas the reference impression states "No significant interval change from
  prior x-ray."

(f) Omitting a comparison detailing a change from a prior study: 0.9958762886597938.
  Candidate omits the comparison statement that "Scapholunate widening [is]
  slightly increased from prior."

----------------------------------


============================================================
Analyzing column: clean_degen
============================================================
[VERBOSE] __call__ invoked with 485 references and 485 hypotheses
Processing data...making prompts
[VERBOSE] Dataset created with 485 examples
Map: 100%|████████████████████████████████████████| 485/485 [00:00<00:00, 45364.56 examples/s]
[VERBOSE] Prompts generated
Done.
[VERBOSE] Starting inference...
[VERBOSE] _infer_azure started
[VERBOSE] Processing 485 examples in 1 batches (batch_size=512)
==== Beginning Inference ====
0it [00:00, ?it/s][VERBOSE] Processing batch 1/1...
[VERBOSE] Sending 485 requests in parallel...
Parallel requests: 100%|██████████████████████████| 485/485 [00:15<00:00, 31.80it/s]
[VERBOSE] All 485 parallel requests completed
[VERBOSE] Batch 1 completed. Total completions: 485
1it [00:16, 16.26s/it]
==== End Inference ====
Computing summary ...
Seconds per example:  0.07308227008151025

[clean_degen] Mean GREEN score: 0.7682 ± 0.2635

-------------gpt-5.1----------------
 [Summary]: Green average 0.7682228310063361 and standard deviation 0.2635297006596275
 [Clinically Significant Errors Analyses]: <accuracy>. <representative error>

(a) False report of a finding in the candidate: 0.6268041237113402.
  Candidate reports "Mild osteoarthritic change of the radiocarpal and intercarpal
  joint spaces," which is not present in the reference and represents a clinically
  relevant degenerative finding at a different joint.

(b) Missing a finding present in the reference: 0.6515463917525773.
  Omits "Mild-to-moderate degenerative changes of multiple PIP and DIP joints"

(c) Misidentification of a finding's anatomic location/position: 0.8515463917525773.
  Degenerative change is localized to the first carpometacarpal joint in the
  reference but is instead described in the radiocarpal, intercarpal, and
  interphalangeal joints in the candidate.

(d) Misassessment of the severity of a finding: 0.8123711340206186.
  Reference describes "mild" degenerative changes, while the candidate describes
  "mild to moderate narrowing" of the first carpometacarpal joint, altering the
  reported severity of degenerative disease.

(e) Mentioning a comparison that isn't in the reference: 1.0.
  None – reference report not provided for comparison.

(f) Omitting a comparison detailing a change from a prior study: 0.9855670103092784.
  (No prior study comparison is described in the reference.)

----------------------------------


============================================================
Analyzing column: clean_degen_excl_nofindings
============================================================
[VERBOSE] __call__ invoked with 485 references and 485 hypotheses
Processing data...making prompts
[VERBOSE] Dataset created with 485 examples
Map: 100%|████████████████████████████████████████| 485/485 [00:00<00:00, 50009.52 examples/s]
[VERBOSE] Prompts generated
Done.
[VERBOSE] Starting inference...
[VERBOSE] _infer_azure started
[VERBOSE] Processing 485 examples in 1 batches (batch_size=512)
==== Beginning Inference ====
0it [00:00, ?it/s][VERBOSE] Processing batch 1/1...
[VERBOSE] Sending 485 requests in parallel...
Parallel requests: 100%|██████████████████████████| 485/485 [00:24<00:00, 19.94it/s]
[VERBOSE] All 485 parallel requests completed
[VERBOSE] Batch 1 completed. Total completions: 485
1it [00:25, 25.34s/it]
==== End Inference ====
Computing summary ...
Seconds per example:  0.08450498531774148

[clean_degen_excl_nofindings] Mean GREEN score: 0.7651 ± 0.2535

-------------gpt-5.1----------------
 [Summary]: Green average 0.7650736377025037 and standard deviation 0.253543549746443
 [Clinically Significant Errors Analyses]: <accuracy>. <representative error>

(a) False report of a finding in the candidate: 0.5979381443298969.
  Candidate reports "Mild osteoarthritic change of the radiocarpal and intercarpal
  joint spaces," which is not present in the reference.

(b) Missing a finding present in the reference: 0.6515463917525773.
  Omission of mild-to-moderate degenerative changes of multiple PIP and DIP joints.

(c) Misidentification of a finding's anatomic location/position: 0.843298969072165.
  Degenerative/arthritic changes are shifted from the thumb interphalangeal joint
  (reference) to the right fourth and fifth distal interphalangeal joints (candidate).

(d) Misassessment of the severity of a finding: 0.8144329896907216.
  Reference describes "mild-to-moderate" degenerative changes, while the candidate
  describes only "mild" osteoarthritic change, underestimating severity.

(e) Mentioning a comparison that isn't in the reference: 1.0.
 None

(f) Omitting a comparison detailing a change from a prior study: 0.9876288659793815.
  Candidate fails to mention that the mild widening of the scapholunate interval
  is unchanged from prior, which is explicitly stated in the reference.

----------------------------------


Saving results to: green_results.csv
Done!

============================================================
Summary Statistics
============================================================
answer_gt: mean=0.9938, std=0.0785, n=485
base: mean=0.7940, std=0.2491, n=485
clean_degen: mean=0.7682, std=0.2638, n=485
clean_degen_excl_nofindings: mean=0.7651, std=0.2538, n=485
```

## Key Observations

1. **answer_gt vs answer_gt**: Near-perfect score (0.9938) as expected when comparing ground truth to itself
2. **base**: Score of 0.7940 with errors mainly in false reports and missing findings
3. **clean_degen**: Score of 0.7682, similar error patterns
4. **clean_degen_excl_nofindings**: Score of 0.7651, lowest among the comparison columns

## Performance Notes

- With `batch_size: 512` and `parallel: true`, 485 examples processed in ~8-25 seconds per column
- Parallel requests achieve ~20-56 requests/second depending on response complexity
- Total time for 4 columns × 485 examples ≈ 1-2 minutes
