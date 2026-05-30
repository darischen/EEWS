# Sentiment-Enhanced Transformer: Deep Problem Analysis & Solution Plan

## Part 1: The Core Problem - Why the Model Overfits

### Root Cause: Distribution Shift Across Time Periods

The transformer was trained on data spanning **1962-2026**, but the market regimes are fundamentally different across decades:

**Training Data (1962-2006):**
- Stable, pre-crisis economy
- Fixed exchange rates, different inflation regimes
- Tech bubble formation but no collapse yet
- Patterns: slow, trending, mean-reverting
- Market psychology: different risk appetite

**Validation Data (2006-2016):**
- Post-2008 financial crisis aftermath
- High volatility, regime uncertainty
- Quantitative easing era
- Patterns: volatile, regime-switching, flash crashes
- Market psychology: risk aversion, central bank dependence

**Test Data (2016-2026):**
- Low-volatility "TINA" (There Is No Alternative) era
- Fed tightening followed by pivot
- Pandemic shock and recovery
- Tech boom and correction
- Patterns: completely different from training era

### Why This Causes the Observed Behavior

The model **memorizes training patterns** (1962-2006):
- Learns: "When volume increases + price up → predict further up"
- Learns: "Seasonality drives returns"
- Learns: "Mean reversion over 20 days"

Then when applied to validation (2006-2016):
- These patterns **don't work** in a post-crisis, low-rate environment
- Model struggles → **validation loss spikes to 1.28, 1.79**
- The model can't generalize because **the game itself changed**

### Evidence from Training Logs

| Epoch | Train | Val | Test | Interpretation |
|-------|-------|-----|------|-----------------|
| 1 | 2.63 | 0.29 | 0.20 | Model learning from training data |
| 3 | 0.62 | 0.03 | 0.06 | ✅ Best val (epoch 3 checkpoint saved) |
| 4-6 | 0.05 | 1.28 | 0.14 | ⚠️ **REGIME SHIFT**: Train loss ↓ but val loss ↑↑↑ |
| 7 | 0.045 | 0.19 | 0.08 | Briefly recovers, but unstable |
| 12-18 | 0.04 | 0.79→1.79 | 0.28→0.99 | **Volatility escalates** |

**The gap is massive:** Train=0.04 vs Val=1.79 = **44x difference**

This isn't normal overfitting (where val loss just stays high). This is **regime rejection**—the model learned one market's rules and can't adapt when the rules change.

---

## Part 2: Why Sentiment Solves This (In Theory)

### The Key Insight: Sentiment is Regime-Invariant

Sentiment captures **"what market players believe today"** rather than **"what worked in 1980"**.

**Example:**
- 1980: Stock up 5% = "strong fundamentals, buy more"
- 2008: Stock up 5% = "relief rally, sell on strength" (opposite strategy!)
- 2020: Stock up 5% = "Fed support, buy the dip"

But sentiment today would say:
- Positive sentiment (80%) → people are optimistic → prices should sustain
- Negative sentiment (20%) → people are scared → prices should fall

**Sentiment is a bridge between regimes** because it captures current market psychology, not historical patterns.

### How Sentiment Helps Generalization

**Without sentiment (current):**
- Model sees: `[price_up, volume_up, volatility_down]`
- Learned association from 1980s: "This means buying opportunity"
- Applied to 2010: "This means relief rally, not sustainable"
- **Result:** Conflicts with validation patterns → high loss

**With sentiment (proposed):**
- Model sees: `[price_up, volume_up, volatility_down, sentiment=+0.7]`
- Model learns: "When sentiment is positive AND price patterns align → sustainable"
- When sentiment contradicts (e.g., `sentiment=-0.6`): "Ignore the old pattern, sentiment says risk-off"
- **Result:** Sentiment provides a "regime selector" that helps the model adapt

### Analogy: Adding a "Market Mood" Channel

Think of it like a human trader:
- **Old way (price-only):** "I learned in 2000 that tech rallies mean opportunity"
  - In 2008: "Why is tech crashing even though it's rallying?!" (overfits to old pattern)
- **New way (with sentiment):** "Today's sentiment is -80% (fear)"
  - "Ah, this rally is bear trap, not opportunity" (uses sentiment to override old pattern)

---

## Part 3: Why Sentiment Works Specifically for Transformers

Transformers excel with **multi-modal, time-aligned signals** because of attention mechanisms:

### Transformer Attention Advantage

```
LSTM approach:
- New price input → hidden state update → struggles to "unlearn" old pattern
- Limited capacity to weight different signals differently

Transformer approach:
- Attention mechanism can learn:
  * "When sentiment is positive, weight price momentum heavily"
  * "When sentiment is negative, weight volatility heavily"
  * "In mixed sentiment, use both signals with different weights"
- Each time step can dynamically reweight features based on context
```

**Key benefit:** Transformers can learn **context-dependent feature importance**
- With LSTM: "Always trust volume" (fixed)
- With Transformer + Sentiment: "Trust volume IF sentiment supports it" (dynamic)

This is exactly what we need to handle regime shifts!

---

## Part 4: The Implementation Strategy

### Why We Need Multi-Source Sentiment

**Problem with single sources:**
- NewsAPI: Only ~30 days historical, single perspective (news articles)
- Need 40+ years of data to capture regime transitions

**Solution:** **Multi-Signal Sentiment from Finnhub + Reddit**

```
1. Finnhub (Historical News Sentiment):
   └─ 40+ years of historical news sentiment data
   └─ Institutional perspective (news articles)
   └─ Company-specific and sector trends

2. Reddit (Social Sentiment):
   └─ Retail trader psychology and sentiment
   └─ Captures crowd sentiment (r/stocks, r/investing, r/wallstreetbets)
   └─ Complements institutional view

3. Combine signals into composite sentiment index
   └─ Divergence between news and social sentiment = regime change indicator
   └─ Both agree = strong signal

4. Use full historical sentiment (1980-2026) for better training
   └─ Model can learn how sentiment evolved across market regimes
   └─ Transfer learning with historical data is more robust
```

### Why This Works

The model already understands **general price dynamics** (from 1962-2026 training). What it's missing is **"how to use external signals to adapt across regimes"**.

By fine-tuning on recent data with sentiment:
- Sentiment tells the model: "This is what current market mood is"
- Model learns: "Adjust my 1962-based patterns based on TODAY'S sentiment"
- Result: Generalizes better to unseen future data because sentiment provides a "regime detector"

---

## Part 5: Implementation Plan

### Phase 1: Sentiment Data Collection (Multi-Source)

**Objective:** Create historical sentiment time series for all tickers (1980-2026)

**Steps:**

#### 1. Finnhub Historical News Sentiment
1. Query Finnhub API for company news and sentiment
   - Coverage: 40+ years of historical data
   - Provides pre-computed sentiment scores from FactSet
   - Rate limit: Free tier allows reasonable requests
   - Batching: Process all tickers, cache results

2. Extract Finnhub sentiment
   - Finnhub provides `sentiment` score directly (pre-computed)
   - No need to extract from article text
   - Aggregates to daily sentiment per ticker

#### 2. Reddit Sentiment (Social Signal)
1. Query Reddit using PRAW (Python Reddit API Wrapper)
   - Subreddits: r/stocks, r/investing, r/wallstreetbets
   - Post dates: Collect historical threads mentioning tickers
   - Extract sentiment from post titles and comments

2. Compute Reddit sentiment
   - Use FinBERT for financial text sentiment analysis
   - Why FinBERT: Trained on financial language, better than TextBlob
   - Aggregate to daily sentiment per ticker

#### 3. Composite Sentiment Index
1. Normalize both sources to [-1, 1] scale
2. Combine: `composite_sentiment = 0.6 * finnhub + 0.4 * reddit`
   - Weight Finnhub more (institutional, verified data)
   - Weight Reddit as secondary (crowd psychology)

3. Divergence metric: `abs(finnhub - reddit)` indicates regime uncertainty
   - When sources disagree = potential regime change

**Output:** `data/sentiment/daily_sentiment.csv` with columns:
- `[ticker, date, finnhub_sentiment, reddit_sentiment, composite_sentiment]`

### Phase 2: Data Preparation (1.5 hours)

**Objective:** Extend sequences to include sentiment as 6th feature

**Steps:**
1. Load price sequences (existing ChunkedSequenceDataset)
2. Load sentiment data
3. For each sequence in (Feb-Mar 2026):
   - Extract corresponding sentiment values (aligned by date)
   - Append to sequence: `[open, high, low, close, volume, sentiment]`
   - Handle missing sentiment: forward-fill or use 0 (neutral)

4. Normalize sentiment to same scale as OHLCV
   - Currently: OHLCV is StandardScaled (mean=0, std=1)
   - Sentiment: already normalized to [-1, 1]
   - Either: rescale to [-1, 1] or StandardScale alongside OHLCV

**Output:** Extended sequences with 6 features instead of 5

### Phase 3: Model Architecture Update (30 min)

**Objective:** Modify transformer to accept 6 input features

**Changes needed:**
```python
# Current
embedding = nn.Linear(5, d_model)  # OHLCV only

# New
embedding = nn.Linear(6, d_model)  # OHLCV + sentiment
```

**Additional considerations:**
- Attention heads don't change (still process d_model dimensions)
- Output heads don't change (still predict 4 tasks)
- Just the initial embedding layer expands from 5→6

**Why this works:** Attention mechanism is agnostic to feature count, only cares about sequence length and embedding dimension.

### Phase 4: Fine-tuning (2-3 hours)

**Objective:** Train existing checkpoint to use sentiment signal

**Strategy:**
1. Load best checkpoint (epoch 3: val_loss=0.0276)
2. Resize embedding layer: 5→6 dimensions
   - Initialize 6th dimension randomly or copy from 5th
3. Freeze most of transformer, train only:
   - Embedding layer (learn feature importance)
   - Top 1-2 transformer layers
   - Output heads
4. Use low learning rate (1e-5 instead of 5e-4)
   - Preserve learned patterns, adapt slowly
5. Train on Feb-Mar 2026 data (8-12 epochs)
   - Small dataset (~1-2 months) but aligned data
6. Validate on April 2026+ (unseen recent data)

**Expected metrics:**
- Val loss should be **more stable** than before (less spiking)
- Val loss might be **higher absolute value** (different regime) but more consistent
- Test performance on unseen April+ should improve

### Phase 5: Validation (1 hour)

**Objective:** Verify sentiment actually helps

**Tests:**
1. Ablation: Remove sentiment feature, does validation loss increase?
2. Attention analysis: Do attention weights respond to sentiment values?
3. Out-of-sample: Test on May 2026+ data (completely unseen)

---

## Part 6: Why This Might Work (And Failure Modes)

### Why It Should Work
1. ✅ Sentiment provides a regime-adaptive signal
2. ✅ Transformers excel at weighting multi-modal inputs
3. ✅ Transfer learning preserves learned patterns while adapting
4. ✅ Recent data + sentiment are aligned (same Feb-Mar period)

### Why It Might Fail
1. ❌ Sentiment might just add noise (not predictive)
   - Counter: Human traders DO use sentiment
2. ❌ Feb-Mar 2026 is too short to learn sentiment weighting
   - Counter: Transfer learning means we're not learning from scratch
3. ❌ NewsAPI coverage might be incomplete (missing tickers)
   - Counter: Use most-traded tickers, ignore sparse coverage
4. ❌ TextBlob sentiment might be too coarse
   - Counter: Could upgrade to FinBERT later if needed

### Success Criteria
- [ ] Validation loss volatility **decreases** (fewer spikes)
- [ ] Validation loss **stabilizes** around 0.15-0.25 (not 1.79)
- [ ] Out-of-sample test loss on unseen data **improves**
- [ ] Attention visualization shows sentiment weighting changes across batches

---

## Part 7: Timeline & Dependencies

**Total estimated time: 6-8 hours**

| Phase | Time | Dependencies |
|-------|------|--------------|
| 1: Sentiment Collection | 4h | Finnhub API key, Reddit API (PRAW) |
| 2: Data Prep | 1.5h | Phase 1 complete |
| 3: Architecture | 0.5h | None (quick code change) |
| 4: Fine-tuning | 3h | Phases 1-3 complete |
| 5: Validation | 1h | Phase 4 complete |

**Blocker risks:**
- Finnhub rate limits → Cache results, batch requests
- Reddit API data sparse for older dates → Fill with Finnhub data
- FinBERT model download → One-time, then cached
- GPU memory issues with 8-10 dimensional sequences → Monitor during training

---

## Decision Point

**Should we proceed?**

This approach assumes:
1. Sentiment is actually predictive of price movements
2. Transformer attention can learn to weight sentiment dynamically
3. Recent data is representative enough for transfer learning

**If yes:** Start with Phase 1 (collect sentiment data)
**If no:** Alternative approaches:
- Train separate model on recent data only (but much smaller dataset)
- Use ensemble (historical model + recent-only model)
- Add other signals (VIX, bond yields, USD index) instead of sentiment
