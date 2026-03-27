# Heuristic Ensemble Flow

This note describes the current heuristic ensemble flow graphically and at a high level.

## Dataset Summary

Total samples: `2248`

### Split Counts

| Split | Samples |
|---|---:|
| Train | 1797 |
| Val | 224 |
| Test | 227 |

### Per-Dataset Composition

| Dataset | Train | Val | Test |
|---|---:|---:|---:|
| CVC-300 | 48 | 6 | 6 |
| CVC-ClinicDB | 489 | 61 | 62 |
| CVC-ColonDB | 304 | 38 | 38 |
| ETIS-LaribPolypDB | 156 | 19 | 21 |
| Kvasir | 800 | 100 | 100 |

## Individual Model Test Results

These are the test results used as the main reference for comparing the standalone models with the heuristic ensemble tool.

| Method | Dice | IoU |
|---|---:|---:|
| UNet | 0.8552 | 0.7553 |
| UNet++ | 0.8697 | 0.7750 |
| UNetV2  | 0.8101 | 0.6915 |
| DeepLabV3+ | 0.8651 | 0.7680 |
| CollabSeg | 0.8641 | 0.7977 |

### Current Heuristic Tool Result

The current heuristic ensemble tool  achieved:

- Dice: `0.8641`
- IoU: `0.7977`


### CollabSeg Per-Dataset Results

| Dataset | Dice | IoU |
|---|---:|---:|
| CVC-300 | 0.9458 | 0.8987 |
| CVC-ClinicDB | 0.9016 | 0.8432 |
| CVC-ColonDB | 0.8828 | 0.8198 |
| ETIS-LaribPolypDB | 0.7579 | 0.6802 |
| Kvasir | 0.8513 | 0.7797 |
|Our tool | 0.8641
## Overall Flow

```text
                  +----------------------+
                  |  Input Image + Prompt |
                  +----------+-----------+
                             |
                             v
                +---------------------------+
                | Run All Segmentation      |
                | Models Independently      |
                | - UNet                    |
                | - UNet++                  |
                | - UNetV2                  |
                | - DeepLabV3+              |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Candidate Outputs         |
                | - one mask per model      |
                | - one probability map     |
                |   per model               |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Heuristic Assessment      |
                | for Each Model Output     |
                | - confidence              |
                |   foreground + background |
                | - agreement               |
                |   overlap with other      |
                |   model outputs           |
                | - shape                   |
                |   area plausibility +     |
                |   connectivity            |
                | - boundary                |
                |   contour consistency     |
                |   with image edges        |
                | - prompt consistency      |
                |   size/location match     |
                |   to prompt hints         |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Final Score Per Model     |
                | Rank Models from Best     |
                | to Weakest for this image |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Decision Policy           |
                | - consensus               |
                | - fuse_top_k              |
                | - select_best             |
                +-------------+-------------+
                              |
                              v
                +---------------------------+
                | Final Output              |
                | - final segmentation mask |
                | - decision mode           |
                | - selected model(s)       |
                | - short reason            |
                +---------------------------+
```

## Model-Level Flow

For each image, the current tool behaves like this:

```text
Image
  |
  +--> UNet -----------+
  |                    |
  +--> UNet++ ---------+--> compare all outputs --> score all outputs --> decide final result
  |                    |
  +--> UNetV2 ---------+
  |                    |
  +--> DeepLabV3+ -----+
```

So the current system does not trust one model first.
It first gathers all candidate results, then reasons over them.

## Scoring Stage

Each model output is converted into 5 final scoring terms.

These 5 terms are the only ones used in the final weighted decision.

### Final Score Formula

```text
Final Score =
0.35 * Confidence Score
+ 0.25 * Agreement Score
+ 0.20 * Shape Score
+ 0.15 * Boundary Score
+ 0.05 * Prompt Score
```

## Score Table

| Final Score Term | Meaning | Weight in Final Score | Built From |
|---|---|---:|---|
| Confidence Score | How confident the model is in foreground and background | 0.35 | Foreground confidence + background confidence |
| Agreement Score | How much this model agrees with the other models | 0.25 | Average overlap with the other model masks |
| Shape Score | Whether the predicted object looks structurally plausible | 0.20 | Area plausibility + connected component plausibility |
| Boundary Score | Whether the predicted contour aligns well with image edges | 0.15 | Boundary strength |
| Prompt Score | Whether the prediction matches the user prompt | 0.05 | Prompt-based size/location consistency |

## Score Composition Table

| Final Score Term | Composition |
|---|---|
| Confidence Score | `0.7 * foreground_confidence + 0.3 * background_confidence` |
| Agreement Score | currently equal to `agreement_iou` |
| Shape Score | `0.6 * area_score + 0.4 * components_score` |
| Boundary Score | normalized version of `boundary_strength` |
| Prompt Score | neutral if prompt is generic, adjusted if prompt contains size/location hints |

## Shape Score Breakdown

```text
Mask
  |
  +--> Area check ----------> Area Score
  |
  +--> Connectivity check --> Components Score
                               |
                               v
                     Shape Score = combine both
```

So shape is not one raw measurement.
It is a summary of:

- whether the segmented region size looks reasonable
- whether the segmented region is clean and coherent

## Agreement Score Breakdown

```text
UNet++ mask
  |
  +--> compare with UNet
  +--> compare with UNetV2
  +--> compare with DeepLabV3+
          |
          v
   average all overlaps
          |
          v
   Agreement Score
```

So agreement measures how much support one model gets from the rest of the ensemble.

## Boundary Score Breakdown

```text
Predicted Mask
  |
  +--> detect outer contour of the mask
  |
  +--> compare contour against image edges
  |
  +--> produce boundary strength
  |
  +--> normalize into boundary score
```

So boundary score is meant to answer:

"Does the predicted contour sit on a meaningful visual boundary in the image?"

## Decision Policy

After all models are scored, the system applies one of three policies.

### 1. Consensus

Use this when the models strongly agree overall.

Handling:
- combine the outputs from all models
- return one consensus mask

Interpretation:
- the ensemble is acting as a group

### 2. Fuse Top Results

Use this when the top-ranked models are very close in score and also compatible.

Handling:
- take the top few model outputs
- combine them into one final result

Interpretation:
- a few strong models collaborate

### 3. Select Best

Use this when one model clearly stands out.

Handling:
- return the output of the top-ranked model directly

Interpretation:
- one model wins for that image

## Policy Flow

```text
Ranked Model Outputs
        |
        v
Are all models strongly consistent?
        |
   +----+----+
   |         |
  Yes       No
   |         |
   v         v
Consensus   Are top models close and compatible?
                |
           +----+----+
           |         |
          Yes       No
           |         |
           v         v
      Fuse Top K   Select Best
```

## Final Output

No matter which policy is used, the system returns one final mask for the image.

That final mask may come from:

- one single model
- a fusion of top models
- a consensus of all models

So the tool itself should be treated as one complete inference method when compared against the standalone models.
