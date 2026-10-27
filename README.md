<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayesian and Neural Approaches for Chord Generation in Pop Music

> **Example project from Autumn 2025.** This is a copy of [Michellexpeng/Bayesian-Project](https://github.com/Michellexpeng/Bayesian-Project), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

How much of a pop song's chord progression can a model predict from the chords before it, and does it help to know the key, to reduce chords to their harmonic function, or to let hidden states and durations into the model? The team answered this on POP909, a corpus of 909 pop songs with beat-aligned chord and key annotations, with four probabilistic models, and separately trained an LSTM to generate piano accompaniment from a melody.

## Summary

Chords are first aligned to beats and transposed, then reduced from 193 absolute chord symbols to 20 functional labels, Roman-numeral degrees relative to the song's key. Four models predict the next chord: a Markov chain over absolute chords, a Markov chain over functional chords with separate transition matrices for major and minor keys, and two latent-state models fit by Gibbs sampling, a Bayesian HMM with a key-aware prior and a variant that models how long each hidden state lasts. The functional, mode-conditioned chain cuts test perplexity from 11.4 to 5.0 and raises next-chord accuracy from 34 to 42 percent. The latent-state models reach 46 and 48 percent accuracy on their own test split. All four generate chord progressions as MIDI. A second, separate task trains an encoder-decoder LSTM to produce a bar of piano accompaniment from the melody and the preceding bars, and writes the results to MIDI.

## Team and contributions

Team "Regularize This!", Autumn 2025.

- **Michelle Peng** ([@Michellexpeng](https://github.com/Michellexpeng)): the POP909 parser, chord preprocessing and functional-harmony features, the baseline and mode-conditional Markov chains, the evaluation and comparison scripts, the MIDI generation, the EDA and visualization notebooks.
- **Luna Lu** ([@uchilunalu](https://github.com/uchilunalu)): the Bayesian HMM and the duration-modeling variant, their Gibbs samplers, training, evaluation and generation scripts.
- **Wen Yao** ([@weckiness0807](https://github.com/weckiness0807)): the LSTM accompaniment model, its data pipeline and MIDI export, the LSTM section of the write-up, and the write-up's final assembly.

The project ran from November 11 to December 10, 2025. The attributions above follow the [commit history](https://github.com/uchicago-dsi-bayesianml/au25-chord-progression-hmm/commits/main).

## Getting started

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/music-x-lab/POP909-Dataset.git /tmp/pop909 && mkdir -p data && cp -r /tmp/pop909/POP909 data/POP909
```

Train and evaluate the two Markov chains, then compare them:

```sh
python scripts/train_baseline.py --pop909 data/POP909 --out models/hmm_baseline.pkl
python scripts/train_conditional.py --pop909 data/POP909 --out models/hmm_conditional.pkl
python scripts/test_baseline.py --model models/hmm_baseline.pkl --pop909 data/POP909
python scripts/test_conditional.py --model models/hmm_conditional.pkl --pop909 data/POP909
python scripts/compare_models.py
```

Train and evaluate the latent-state models, which take a few minutes each:

```sh
python scripts/train_hdp_hmm.py
python scripts/train_hdp_hsmm.py
python scripts/compare_hdp.py --pop909 data/POP909
```

Generate a chord progression and open it in any MIDI player:

```sh
python music_generation/generate_full_music.py --model models/hmm_conditional.pkl --mode major --bars 8 --output generated_music/song.mid
```

Trained models are included in `models/`, so generation works without training. The LSTM lives in `LSTM/LSTM.ipynb`, written for Google Colab with a GPU and the dataset on Google Drive; its preprocessing cells convert POP909 MIDI files to bar-level piano rolls. The guides `MODEL_TRAINING_GUIDE.md` and `music_generation/MUSIC_GENERATION_GUIDE.md` have more options.

## Data

Data files are never committed to this repository. Git ignores `data/`.

- **Source:** [POP909](https://github.com/music-x-lab/POP909-Dataset), Wang et al., ISMIR 2020: 909 Chinese pop songs as MIDI with aligned melody, bridge and piano tracks, beat positions, key and chord annotations.
- **Terms:** MIT license.
- **How to get it:** clone the dataset repository as shown above.
- **Where to put it:** `data/POP909/`, one numbered folder per song.

The EDA notebook also looks at a Kaggle collection of classical piano MIDI files, which the proposal planned as a second corpus. It isn't used by any model.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: both tasks, models, results, challenges, future work. |
| `src/data/` | POP909 parser, chord normalization and transposition, Roman-numeral features, the dataset class with the key-aware prior. |
| `src/models/hdp_hmm.py`, `hdp_hsmm.py` | The two latent-state models and their Gibbs samplers. |
| `scripts/` | Training, testing and comparison scripts for all four models. |
| `models/` | The four trained models as pickles. |
| `music_generation/` | Chord-progression and full-arrangement MIDI generation, with guides. |
| `notebooks/` | EDA of POP909 and the classical set, and the figures: transition heatmaps, vocabulary, perplexity comparisons. |
| `validation_results/` | Detailed reports on the two Markov chains. |
| `LSTM/` | The accompaniment notebook and ten generated MIDI files, five of them with humanized timing. |
| `Bayesian Conditional Harmonization on Symbolic Music.docx` | The project proposal. |

## Method

**Representation.** Chord annotations are sampled at each beat. For the baseline, chords are transposed to C major or A minor and keep their quality, giving 193 symbols. For the other models, each chord becomes a scale degree relative to the song's key, with the mode kept as a separate variable. That gives 20 labels: the seven major-key degrees, four minor-key degrees, eight chromatic degrees that fall outside the scale, and "no chord".

**Markov chains.** The baseline is a first-order chain over the 193 symbols with add-one smoothing. The conditional model is the same chain over the 20 functional labels with one transition matrix for major keys and one for minor. Both are fit by counting. Prediction is the most likely next chord given the previous one.

**Latent-state models.** Both have 24 hidden states, a categorical emission distribution per state over 425 labels, scale degree combined with the full chord quality and inversion, Dirichlet priors on transitions and emissions, and a key-aware prior that adds pseudo-counts to dominant-to-tonic and subdominant-to-dominant transitions. They are fit with 50 sweeps of Gibbs sampling: forward-filtering backward-sampling for the state sequences, conjugate Dirichlet draws for the parameters. The first model is a standard HMM. The second separates the transition matrix between distinct states from how long a state persists, puts a Gamma prior on each state's mean duration, and samples Poisson durations when generating. Evaluation uses the forward algorithm, so next-chord prediction comes from the filtered belief over hidden states, not from the last state alone.

**LSTM.** An encoder LSTM reads the melody and piano of the preceding bars; a decoder LSTM produces the current bar's piano roll step by step from the melody and its own previous output, with teacher forcing in training. Binary cross-entropy with positive weighting handles the sparse 128-pitch targets, and a threshold plus top-k selection controls density at generation time. The notebook also tries a small Transformer.

### What is Bayesian here

The two Markov chains are maximum-likelihood models with a smoothing prior; their value is in the representation, not the inference. The latent-state models are the Bayesian part. Every transition row and emission row has a Dirichlet prior, the key-aware pseudo-counts encode music theory as prior belief, and Gibbs sampling draws from the joint posterior over hidden state sequences and parameters, alternating a conjugate update for the parameters with an exact sample of the state path given them. That is the course's HMM material and its MCMC material in one model. The duration variant adds a Gamma-Poisson conjugate pair for how long a hidden state lasts. The LSTM is the non-Bayesian contrast: a point-estimate model that gives probabilities per pitch but no posterior over anything.

## Results

Reported by the authors on POP909 test songs. The first two rows share a random 70/15/15 split with seed 42; the last two use the first 70 percent of songs for training and the last 15 percent for testing, and a 425-label vocabulary. See the caveat below.

| Model | Test perplexity | Next-chord accuracy |
|---|---|---|
| Markov chain, 193 absolute chords | 11.44 | 34.0% |
| Markov chain, 20 functional chords, by mode | 4.96 | 41.8% |
| Bayesian HMM, 24 states, key-aware prior | 9.11 | 46.2% |
| Same, with explicit durations | 8.75 | 48.1% |

The functional, mode-conditioned chain is the clearest result: the representation halves perplexity. The authors' error analysis shows that its mistakes are mostly functionally similar substitutions. The transition heatmaps in `notebooks/Model_Visualization.ipynb` are readable music theory: strong I, IV, V and vi structure in major keys, i, iv, v and VI in minor.

The LSTM results are qualitative: piano-roll plots at different decoding thresholds and ten generated MIDI files. Early models collapsed to repeating one chord; context length, loss weighting and a sparsity penalty fixed that.

## What to learn from this project

Course staff added this section. The core work is sound and the code is the team's own. The points below are about naming things precisely and comparing fairly.

- **Name the model you built.** The write-up calls the latent-state models "HDP-HMM" and "HDP-HSMM" and says they infer the number of states from an infinite state space. The code is a finite 24-state Bayesian HMM with Dirichlet priors. There is no hierarchical Dirichlet process. The duration variant uses geometric durations in inference, which makes it an HMM with a different parametrization, and only draws Poisson durations when generating. Both are good models and worth presenting as what they are.
- **Compare on the same test set.** The two Markov chains and the two latent-state models use different splits and different vocabularies, so the four rows above aren't one comparison. Within each pair the comparison is fair. One split, fixed at the start, would have made the whole table meaningful, including the interesting question of whether hidden states beat an observed-state chain at all.
- **Describe the vocabulary you have.** The report says the 20 labels are seven major, seven minor, five chromatic and one special, and that chromatic chords such as bII and bVII are handled. The labels in the trained conditional model are seven major degrees, four minor degrees, eight unnamed chromatic degrees written as question marks, and "no chord". The report also presents the latent-state models as working on the same functional harmony, but their pickled vocabularies hold 425 labels, scale degree with full quality and inversion, which is why their perplexities sit between the two chains rather than below them. A 48 percent next-chord accuracy over 425 labels is a stronger result than the table makes it look, and the write-up undersells it by not saying so.
- **Say what changed since the proposal.** The proposal promised NumPyro or Pyro, calibrated uncertainty and listening tests. None appear, and the write-up doesn't say so. Scope changes are normal; the write-up should record them.
- **Small things.** Two of the three GitHub accounts had no name set, the write-up names no authors, a citation block said "Your Name", four data files were two-byte placeholders, and the LSTM task is a different task from the chord models, so it is a second project rather than the neural baseline the title suggests.

What the project does well: a clean, modular codebase with reproducible scripts and a fixed seed; a representation decision, functional harmony by mode, that is motivated by music theory and validated by a large perplexity drop; a hand-written Gibbs sampler with forward-filtering backward-sampling that is correct; prediction from the filtered belief rather than a point estimate of the hidden state; interpretable outputs in the form of transition heatmaps and playable MIDI; and an honest account of what the neural model could and couldn't do.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Autumn 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code and notebooks are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.
- Removed four two-byte placeholder files in `LSTM/DATA/` and an empty `Data_files/readme.md`, and removed a citation block with placeholder author text from `MODEL_TRAINING_GUIDE.md`.

## License

The authors have not added a license, so their code is shown here under GitHub's terms for public repositories. Ask them before reusing it.
