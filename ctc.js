/**
 * CTC decoding and forced alignment module.
 * Matches the Python implementation in ear/server.py
 */

const CTCUtils = (function() {
  const BLANK_ID = 0;

  /**
   * Compute log softmax along last dimension
   */
  function logSoftmax(logits, T, vocabSize) {
    const logProbs = new Float32Array(T * vocabSize);
    
    for (let t = 0; t < T; t++) {
      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        const val = logits[t * vocabSize + v];
        if (val > maxVal) maxVal = val;
      }
      
      // Compute exp sum
      let expSum = 0;
      for (let v = 0; v < vocabSize; v++) {
        expSum += Math.exp(logits[t * vocabSize + v] - maxVal);
      }
      const logExpSum = Math.log(expSum);
      
      // Compute log softmax
      for (let v = 0; v < vocabSize; v++) {
        logProbs[t * vocabSize + v] = logits[t * vocabSize + v] - maxVal - logExpSum;
      }
    }
    
    return logProbs;
  }

  /**
   * Viterbi CTC forced alignment.
   * Returns labelAtT: length T array with values:
   *   -1 for blank, or token index 0..n-1 indicating which target token was emitted at frame t.
   * 
   * @param {Float32Array} logProbs - Log probabilities (T x vocabSize)
   * @param {number} T - Number of time frames
   * @param {number} vocabSize - Vocabulary size
   * @param {number[]} targets - Target token IDs
   * @returns {number[]} labelAtT - Frame-level alignment
   */
  function forcedAlignLabels(logProbs, T, vocabSize, targets) {
    const n = targets.length;
    if (n === 0) {
      return new Array(T).fill(-1);
    }

    // Initialize trellis with -Infinity
    // trellis[t][j] = best log-prob of aligning up to frame t with j tokens emitted
    const NEG_INF = -1e10;
    const trellis = [];
    for (let t = 0; t <= T; t++) {
      trellis.push(new Float32Array(n + 1).fill(NEG_INF));
    }
    
    // Base case: at t=0, we've emitted 0 tokens
    trellis[0][0] = 0.0;
    
    // Fill in first column (staying in blank)
    for (let t = 1; t <= T; t++) {
      trellis[t][0] = trellis[t - 1][0] + logProbs[(t - 1) * vocabSize + BLANK_ID];
    }

    // Fill in rest of trellis
    for (let t = 1; t <= T; t++) {
      const blankLogP = logProbs[(t - 1) * vocabSize + BLANK_ID];
      for (let j = 1; j <= n; j++) {
        const targetLogP = logProbs[(t - 1) * vocabSize + targets[j - 1]];
        
        // Stay: previous state was also j tokens emitted (blank transition)
        const stay = trellis[t - 1][j] + blankLogP;
        
        // Change: previous state was j-1 tokens emitted (emit token j)
        const change = trellis[t - 1][j - 1] + targetLogP;
        
        trellis[t][j] = Math.max(stay, change);
      }
    }

    // Backtrack to find alignment
    const labelAtT = new Array(T).fill(-1);
    let j = n;
    
    for (let t = T; t >= 1; t--) {
      if (j > 0) {
        const blankLogP = logProbs[(t - 1) * vocabSize + BLANK_ID];
        const targetLogP = logProbs[(t - 1) * vocabSize + targets[j - 1]];
        
        const stay = trellis[t - 1][j] + blankLogP;
        const change = trellis[t - 1][j - 1] + targetLogP;
        
        if (change > stay) {
          j -= 1;
          labelAtT[t - 1] = j;  // Token index (0-based)
        } else {
          labelAtT[t - 1] = -1;  // Blank
        }
      } else {
        labelAtT[t - 1] = -1;  // Blank
      }
    }

    return labelAtT;
  }

  /**
   * Convert frame labels to token spans.
   * Returns array of [start, end] for each token.
   */
  function spansFromLabels(labelAtT, nTokens) {
    const starts = new Array(nTokens).fill(null);
    const ends = new Array(nTokens).fill(null);
    
    for (let t = 0; t < labelAtT.length; t++) {
      const lab = labelAtT[t];
      if (lab >= 0 && lab < nTokens) {
        if (starts[lab] === null) {
          starts[lab] = t;
        }
        ends[lab] = t;
      }
    }
    
    return starts.map((start, i) => [start, ends[i]]);
  }

  /**
   * Compute spans based on emission midpoints (for UI display).
   * Returns array of [start, end] for each token.
   */
  function spansFromEmits(emitFrames, totalFrames) {
    const n = emitFrames.length;
    const spans = [];
    
    for (let i = 0; i < n; i++) {
      const emitI = emitFrames[i];
      if (emitI === null) {
        spans.push([null, null]);
        continue;
      }
      
      let start;
      if (i === 0) {
        start = 0;
      } else {
        const prev = emitFrames[i - 1];
        start = prev !== null ? Math.floor((prev + emitI) / 2) : 0;
      }
      
      let end;
      if (i === n - 1) {
        end = totalFrames - 1;
      } else {
        const next = emitFrames[i + 1];
        end = next !== null ? Math.floor((emitI + next) / 2) : totalFrames - 1;
      }
      
      if (start > end) {
        [start, end] = [end, start];
      }
      
      spans.push([start, end]);
    }
    
    return spans;
  }

  /**
   * Greedy CTC decoding (collapse repeated tokens, remove blanks).
   */
  function greedyDecode(logProbs, T, vocabSize) {
    const decoded = [];
    let prevToken = -1;
    
    for (let t = 0; t < T; t++) {
      // Find argmax
      let maxIdx = 0;
      let maxVal = logProbs[t * vocabSize];
      for (let v = 1; v < vocabSize; v++) {
        const val = logProbs[t * vocabSize + v];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }
      
      // Collapse repeated and skip blanks
      if (maxIdx !== prevToken && maxIdx !== BLANK_ID) {
        decoded.push(maxIdx);
      }
      prevToken = maxIdx;
    }
    
    return decoded;
  }

  /**
   * Calculate output length after subsampling (matches Python _subsample_lengths).
   * The model has two conv layers that each halve the time dimension.
   */
  function subsampleLength(inputLength) {
    let len = Math.floor((inputLength + 1) / 2);
    len = Math.floor((len + 1) / 2);
    return len;
  }

  return {
    logSoftmax,
    forcedAlignLabels,
    spansFromLabels,
    spansFromEmits,
    greedyDecode,
    subsampleLength,
    BLANK_ID
  };
})();
