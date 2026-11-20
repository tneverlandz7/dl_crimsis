# Part A Fusion Error Analysis

These six test examples use the median-validation runs selected before test inspection: `fusion_bilstm` seed 0, `image_only` seed 0, and text-only `bilstm` seed 0. The first three are cases where fusion is correct while both single-modality models are wrong. The final three are cases where fusion is wrong while both single-modality models are correct. Probabilities are for the `informative` class.

## Fusion helps

| Tweet ID | Official label | Fusion prediction and P informative | Image-only prediction and P informative | Text-only prediction and P informative | Interpretation |
|---|---|---|---|---|---|
| `910736967196954624` | not informative | not informative, 0.474 | informative, 0.707 | informative, 0.654 | The image is a scientific seismogram and map, while the tweet links to an earthquake event page. Each modality alone looks disaster-related, but their combination is closer to general event information than a directly useful crisis update. The case also shows that the agreed label can depend on a narrow definition of informativeness. |
| `912178184455622656` | not informative | not informative, 0.308 | informative, 0.632 | informative, 0.664 | The satellite image and hurricane-related tweet both appear relevant in isolation. Fusion correctly treats the combined post as general storm observation rather than an actionable or situational update. |
| `931281225096089600` | not informative | not informative, 0.434 | informative, 0.555 | informative, 0.827 | The image is a templated “trending story” card and the tweet repeats its sensational headline. Fusion can use the promotional visual style together with the text to reject a post that both single-modality models mistake for informative earthquake content. |

## Fusion fails

| Tweet ID | Official label | Fusion prediction and P informative | Image-only prediction and P informative | Text-only prediction and P informative | Interpretation |
|---|---|---|---|---|---|
| `910694304712097798` | informative | not informative, 0.366 | informative, 0.759 | informative, 0.750 | The image summarises three Category 4 U.S. landfalls and the tweet explicitly describes the 2017 hurricane pattern. Fusion appears to suppress two individually strong signals, possibly because the infographic resembles generic media content rather than an immediate field update. |
| `909802038804008962` | informative | not informative, 0.345 | informative, 0.553 | informative, 0.831 | The infrared hurricane image and the tweet’s location context are both informative. Their combination is nevertheless classified as generic storm imagery, producing a false negative despite both branches being correct alone. |
| `920822232196231168` | informative | not informative, 0.383 | informative, 0.754 | informative, 0.672 | The image and tweet describe birds displaced by wildfire smoke. This less common environmental consequence may be under-represented in training data, causing fusion to reject a valid informative example even though each modality independently detects relevance. |

## Main finding

Fusion improves average macro-F1, but it does not always behave like a vote between the branches. The learned classifier can overturn two agreeing single-modality predictions in either direction. This is beneficial when the joint context reveals generic, promotional, or non-actionable content, but harmful when an unusual informative case is treated as generic disaster imagery. The examples also show why the reported aggregate scores should be accompanied by qualitative inspection and why official labels should be treated as the evaluation target rather than an unambiguous statement of factual usefulness.
