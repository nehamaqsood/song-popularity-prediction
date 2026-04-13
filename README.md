# Song Popularity Prediction

A deep learning model that predicts song popularity scores (0–100) using audio features, built as a final project for DECISION 546Q at Duke Fuqua.

**Team 45:** Chris Chen, Ziling Chen, Krishna Gupta, Sam MacArthur, Neha Maqsood

---

## Background

Streaming platforms like Spotify and Apple Music spend heavily on promoting new releases, typically distributing budgets evenly regardless of a song's potential. This project looks at whether audio features alone can give a reasonable signal for how popular a song will be — and what that's worth financially if deployed at scale.

---

## Dataset

~15,000 songs with the following features: `energy`, `loudness`, `tempo`, `valence`, `danceability`, `acousticness`, `instrumentalness`, `liveness`, `key`, `audio_mode`, `lyric_density`, `time_signature`, `song_duration`, and a popularity score (0–100).

---

## Model

We used a multilayer perceptron (MLP) with ~137,000 trainable parameters. After experimenting with deeper architectures, we found a shallower, wider network generalized better given only 13 input features.

**Architecture:**
- Input: 13 features
- Hidden Layer 1: 384 neurons, Batch Norm, ReLU, Dropout (0.15)
- Hidden Layer 2: 192 neurons, Batch Norm, ReLU, Dropout (0.15)
- Hidden Layer 3: 96 neurons, Batch Norm, ReLU, Dropout (0.10)
- Output: single popularity score

We applied stratified train/test splitting across three popularity bins (0–30, 30–60, 60–100) to address class imbalance, and used a cosine annealing scheduler to avoid local minima during training.

---

## Results

| Metric | Our Model | Best Baseline (Elastic-Net) |
|--------|-----------|--------------------------|
| RMSE | 19.911 | 20.359 |
| R² Score | 0.220 | 0.008 |
| Accuracy (±10 pts) | 46.32% | — |

The model explains roughly 3x more variance than the best classical regression baseline. Prediction is strongest in the medium and high popularity ranges; low-popularity songs remain the hardest to predict accurately, largely due to data imbalance.

---

## Business Case

Applied to a hypothetical platform with a $500M annual marketing budget across 12M releases, the model could shift 70% of spend toward high-potential songs — improving ROI from 1.5x to ~2.06x and generating an estimated $280M in additional revenue. The full calculation is in the notebook.

---

## Files

- `MA_Final.ipynb` — model code, training, evaluation, and business impact analysis
- `Final_Project_Report.pdf` — full written report
- `T45-Final_Deck.pptx` — presentation slides
- `song_data.zip` — dataset

---

## Running the Code

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
unzip song_data.zip
jupyter notebook MA_Final.ipynb
```
