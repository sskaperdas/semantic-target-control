# FAQ

## Is STC a training method?

No. The controller operates after a frozen scorer has produced one finite
candidate window.

## Does STC re-query the scorer?

No. The public controller and paper protocol do not request another window or
retrieve additional candidates during control.

## Is an admissible candidate necessarily factually correct?

No. Admissibility is an operational semantic status. Factual Hit/MRR are
reported separately.

## What does `unknown` mean?

The available metadata is insufficient to classify the candidate as
operationally admissible or violating. It is handled by an explicit penalty.

## What happens when fewer than q admissible candidates are in Top-M?

The controller issues an infeasibility certificate and returns the frozen base
Top-k under the deployment protocol.

## Why are checkpoints and complete candidate windows absent?

They are large and may depend on licensed source data. The repository provides
code, metadata, curated aggregate evidence and reconstruction instructions.
