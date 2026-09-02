/**
 * Pronunciation scoring module.
 * Matches the Python implementation in ear/server.py
 */

const ScoringUtils = (function() {
  // Chinese initials for splitting pinyin
  const INITIALS = [
    'zh', 'ch', 'sh',  // Retroflexes (must come first - longer matches)
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'j', 'q', 'x',
    'r', 'z', 'c', 's',
    'y', 'w'
  ];

  // Minimal pair initials that are commonly confused
  const MINIMAL_PAIR_INITIALS = new Set([
    'zh-z', 'z-zh',
    'ch-c', 'c-ch',
    'sh-s', 's-sh',
    'j-q', 'q-j'
  ]);

  /**
   * Strip tone number from pinyin token.
   */
  function stripTone(token) {
    if (!token) return token;
    if (/[0-9]$/.test(token)) {
      return token.slice(0, -1);
    }
    return token;
  }

  /**
   * Get tone ID (1-5) from pinyin token.
   * Returns null if no valid tone.
   */
  function toneId(token) {
    if (!token) return null;
    const lastChar = token[token.length - 1];
    if (!/[1-5]/.test(lastChar)) return null;
    return parseInt(lastChar, 10);
  }

  /**
   * Split pinyin into initial and final.
   * Example: "zhong" -> ["zh", "ong"]
   */
  function splitInitialFinal(base) {
    for (const ini of INITIALS) {
      if (base.startsWith(ini)) {
        return [ini, base.slice(ini.length)];
      }
    }
    return ['', base];
  }

  /**
   * Check if two finals are an n/ng confusion pair.
   */
  function isNgPair(finalA, finalB) {
    if (finalA.endsWith('n') && finalB.endsWith('ng')) {
      return finalA.slice(0, -1) === finalB.slice(0, -2);
    }
    if (finalB.endsWith('n') && finalA.endsWith('ng')) {
      return finalB.slice(0, -1) === finalA.slice(0, -2);
    }
    return false;
  }

  /**
   * Check if two base forms are minimal pairs.
   */
  function isMinimalPair(baseA, baseB) {
    if (baseA === baseB) return false;
    
    const [iniA, finA] = splitInitialFinal(baseA);
    const [iniB, finB] = splitInitialFinal(baseB);
    
    // Check initial pairs
    if (MINIMAL_PAIR_INITIALS.has(`${iniA}-${iniB}`) || 
        MINIMAL_PAIR_INITIALS.has(`${iniB}-${iniA}`)) {
      return true;
    }
    
    // Check n/ng confusion
    if (isNgPair(finA, finB)) {
      return true;
    }
    
    return false;
  }

  /**
   * Check if two tones are close (2 vs 3).
   */
  function toneClose(toneA, toneB) {
    if (toneA === null || toneB === null) return false;
    const pair = new Set([toneA, toneB]);
    return pair.has(2) && pair.has(3) && pair.size === 2;
  }

  /**
   * Filter frames where blank is not dominant.
   */
  function filterNonblankFrames(probs, T, vocabSize, threshold = 0.6) {
    const filtered = [];
    
    for (let t = 0; t < T; t++) {
      const pBlank = probs[t * vocabSize + 0];
      if (pBlank < threshold) {
        const frame = new Float32Array(vocabSize);
        for (let v = 0; v < vocabSize; v++) {
          frame[v] = probs[t * vocabSize + v];
        }
        filtered.push(frame);
      }
    }
    
    return filtered;
  }

  /**
   * Compute pronunciation status and error type.
   * 
   * @param {string} targetTok - Target pinyin token
   * @param {string} top1Tok - Top predicted token
   * @param {number} pTarget - Probability of target
   * @param {number} targetMargin - Margin over best alternative
   * @param {string|null} topConfusionTok - Top confusion candidate
   * @returns {[string, string|null]} [status, errorType]
   */
  function computeStatus(targetTok, top1Tok, pTarget, targetMargin, topConfusionTok) {
    const baseT = stripTone(targetTok);
    const baseP = stripTone(top1Tok);
    const toneT = toneId(targetTok);
    const toneP = toneId(top1Tok);

    // Correct: high confidence and exact match
    if (pTarget >= 0.85 && targetMargin >= 0.5 && top1Tok === targetTok) {
      return ['correct', null];
    }

    // Determine if it's "close"
    let isClose = false;
    if (pTarget >= 0.5 && pTarget < 0.85) {
      isClose = true;
    }
    if (baseT === baseP && toneClose(toneT, toneP)) {
      isClose = true;
    }
    if (topConfusionTok && isMinimalPair(baseT, stripTone(topConfusionTok))) {
      isClose = true;
    }

    if (isClose) {
      if (baseT === baseP && toneT !== toneP) {
        return ['close', 'tone'];
      }
      if (baseT !== baseP) {
        const [iniT, finT] = splitInitialFinal(baseT);
        const [iniP, finP] = splitInitialFinal(baseP);
        if (isNgPair(finT, finP)) {
          return ['close', 'consonant'];
        }
        if (iniT !== iniP) {
          return ['close', 'consonant'];
        }
        if (finT !== finP) {
          return ['close', 'vowel'];
        }
      }
      return ['close', null];
    }

    // Wrong
    if (baseT === baseP && toneT !== toneP) {
      return ['wrong', 'tone'];
    }

    const [iniT, finT] = splitInitialFinal(baseT);
    const [iniP, finP] = splitInitialFinal(baseP);
    if (isNgPair(finT, finP)) {
      return ['wrong', 'consonant'];
    }
    if (iniT !== iniP) {
      return ['wrong', 'consonant'];
    }
    if (finT !== finP) {
      return ['wrong', 'vowel'];
    }

    return ['wrong', null];
  }

  /**
   * Evaluate pronunciation for a single syllable.
   * Matches Python implementation in server.py lines 379-420
   * 
   * @param {Float32Array} logProbs - Log probabilities for the token's frames (T x vocabSize)
   * @param {number} T - Number of frames
   * @param {number} vocabSize - Vocabulary size
   * @param {string} targetTok - Target pinyin token
   * @param {number} targetId - Target token ID in vocab
   * @param {string[]} vocab - Vocabulary array
   * @param {number} topk - Number of top predictions to consider
   * @returns {object} Evaluation result
   */
  function evaluateSyllable(logProbs, T, vocabSize, targetTok, targetId, vocab, topk = 3) {
    if (T === 0) {
      return {
        target: targetTok,
        predicted: targetTok,
        top1: targetTok,
        status: 'wrong',
        errorType: null,
        pTarget: 0,
        pTop1: 0,
        pBlankAvg: 1,
        top1Margin: 0,
        targetMargin: 0,
        topConfusions: []
      };
    }

    // Convert log probs to probs (span_probs = span.exp())
    const probs = new Float32Array(T * vocabSize);
    for (let i = 0; i < logProbs.length; i++) {
      probs[i] = Math.exp(logProbs[i]);
    }

    // Compute nonblank mass for each frame
    const nonblankMass = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      nonblankMass[t] = Math.max(0, 1.0 - probs[t * vocabSize + 0]);
    }

    // Select frames where nonblank_mass >= 0.05
    // If no frames pass, use the single best frame (matching Python fallback)
    let selectedFrames = [];
    for (let t = 0; t < T; t++) {
      if (nonblankMass[t] >= 0.05) {
        selectedFrames.push(t);
      }
    }
    
    if (selectedFrames.length === 0) {
      // Fallback: use the frame with highest nonblank mass
      let bestFrame = 0;
      let bestMass = nonblankMass[0];
      for (let t = 1; t < T; t++) {
        if (nonblankMass[t] > bestMass) {
          bestMass = nonblankMass[t];
          bestFrame = t;
        }
      }
      selectedFrames = [bestFrame];
    }

    const usedT = selectedFrames.length;

    // Compute average blank probability over selected frames
    let pBlankSum = 0;
    for (const t of selectedFrames) {
      pBlankSum += probs[t * vocabSize + 0];
    }
    const pBlankAvg = pBlankSum / usedT;

    // Compute conditioned nonblank probabilities
    // For each frame: zero out blank, divide by sum of non-blank, then average across frames
    // This matches: cond_nonblank = used_nonblank / denom; avg_probs = cond_nonblank.mean(dim=0)
    const avgProbs = new Float32Array(vocabSize).fill(0);
    
    for (const t of selectedFrames) {
      // Sum of non-blank probs for this frame (denom)
      let denom = 0;
      for (let v = 1; v < vocabSize; v++) {
        denom += probs[t * vocabSize + v];
      }
      denom = Math.max(denom, 1e-8);
      
      // Add conditioned probabilities
      for (let v = 1; v < vocabSize; v++) {
        avgProbs[v] += probs[t * vocabSize + v] / denom;
      }
    }
    
    // Average over selected frames
    for (let v = 1; v < vocabSize; v++) {
      avgProbs[v] /= usedT;
    }

    // Find top-k (excluding blank)
    const scores = [];
    for (let v = 1; v < vocabSize; v++) {
      scores.push({ id: v, prob: avgProbs[v] });
    }
    scores.sort((a, b) => b.prob - a.prob);
    
    const topkIds = scores.slice(0, topk).map(s => s.id);
    const top1Id = topkIds.length > 0 ? topkIds[0] : targetId;
    const top1Tok = vocab[top1Id];
    
    const pTarget = avgProbs[targetId];
    const pTop1 = avgProbs[top1Id];
    
    // Find second best for margin calculation
    let pTop2 = 0;
    for (const id of topkIds) {
      if (id !== top1Id) {
        pTop2 = avgProbs[id];
        break;
      }
    }
    const top1Margin = pTop1 - pTop2;

    // Target margin: pTarget minus best alternative
    let bestAlt = 0;
    for (const id of topkIds) {
      if (id !== targetId) {
        bestAlt = avgProbs[id];
        break;
      }
    }
    const targetMargin = pTarget - bestAlt;

    // Top confusions (excluding target)
    const topConfusions = topkIds
      .filter(id => id !== targetId)
      .slice(0, 2)
      .map(id => vocab[id]);

    const [status, errorType] = computeStatus(
      targetTok,
      top1Tok,
      pTarget,
      targetMargin,
      topConfusions.length > 0 ? topConfusions[0] : null
    );

    return {
      target: targetTok,
      predicted: top1Tok,
      top1: top1Tok,
      status: status,
      errorType: errorType,
      pTarget: Math.round(pTarget * 10000) / 10000,
      pTop1: Math.round(pTop1 * 10000) / 10000,
      pBlankAvg: Math.round(pBlankAvg * 10000) / 10000,
      top1Margin: Math.round(top1Margin * 10000) / 10000,
      targetMargin: Math.round(targetMargin * 10000) / 10000,
      topConfusions: topConfusions
    };
  }

  /**
   * Apply tone sandhi tolerance to evaluation results.
   * This is a post-processing step that marks predictions as correct/close
   * when they match expected sandhi patterns in natural speech.
   * 
   * Sandhi rules handled:
   * 1. Third tone sandhi (3+3 context):
   *    - Tone 2 predicted → CORRECT (natural sandhi)
   *    - Tone 3 predicted → CORRECT (careful/slow speech, also valid)
   *    - Tone 4/1 predicted → WRONG (not accepted)
   * 
   * 2. 不 (bù) sandhi:
   *    - bu4 + tone 4 → bu2 accepted
   *    - bu4 + non-4th → bu4 only (do NOT accept bu2)
   * 
   * 3. 一 (yī) sandhi:
   *    - yi1 + tone 4 → yi2 accepted
   *    - yi1 + tones 1/2/3 → yi4 accepted
   * 
   * @param {object[]} results - Array of evaluation results from evaluateSyllable
   * @param {string[]} targetTokens - Array of target pinyin tokens
   * @returns {object[]} Results with sandhi tolerance applied
   */
  function applySandhiTolerance(results, targetTokens) {
    if (!results || results.length === 0) return results;
    
    const modifiedResults = results.map((r, idx) => ({ ...r }));
    
    for (let i = 0; i < modifiedResults.length; i++) {
      const result = modifiedResults[i];
      const targetTok = targetTokens[i];
      const predictedTok = result.top1;
      
      // Skip if already correct
      if (result.status === 'correct') continue;
      
      // Get target and predicted base/tone
      const targetBase = stripTone(targetTok);
      const predictedBase = stripTone(predictedTok);
      const targetTone = toneId(targetTok);
      const predictedTone = toneId(predictedTok);
      
      // Only check sandhi if base is the same (only tone differs)
      if (targetBase !== predictedBase) continue;
      
      // Get next syllable info (if exists)
      const hasNext = i < targetTokens.length - 1;
      const nextTargetTone = hasNext ? toneId(targetTokens[i + 1]) : null;
      
      let sandhiStatus = null;  // 'correct' or 'close'
      let sandhiType = null;
      let sandhiNote = null;
      
      // Rule 1: Third tone sandhi (3+3 context)
      if (targetTone === 3 && nextTargetTone === 3) {
        if (predictedTone === 2) {
          // Natural sandhi - tone rises to 2
          sandhiStatus = 'correct';
          sandhiType = 'third_tone';
          sandhiNote = 'sandhi';
        } else if (predictedTone === 3) {
          // Careful/slow speech - keeping original tone 3 is also valid
          // This case: target is 3, predicted is 3, but was marked wrong/close
          // due to confidence issues. In sandhi context, 3 is acceptable.
          sandhiStatus = 'correct';
          sandhiType = 'third_tone_careful';
          sandhiNote = 'careful';
        }
        // Tone 4 or 1 in this context = real error, don't modify
      }
      
      // Rule 2: 不 sandhi
      // bu4 before tone 4 → bu2 (ONLY in this context)
      if (targetBase === 'bu' && targetTone === 4 && nextTargetTone === 4 && predictedTone === 2) {
        sandhiStatus = 'correct';
        sandhiType = 'bu_sandhi';
        sandhiNote = 'sandhi';
      }
      
      // Rule 3: 一 sandhi
      if (targetBase === 'yi' && targetTone === 1) {
        // yi1 before tone 4 → yi2
        if (nextTargetTone === 4 && predictedTone === 2) {
          sandhiStatus = 'correct';
          sandhiType = 'yi_sandhi';
          sandhiNote = 'sandhi';
        }
        // yi1 before tones 1, 2, 3 → yi4
        else if (nextTargetTone !== null && [1, 2, 3].includes(nextTargetTone) && predictedTone === 4) {
          sandhiStatus = 'correct';
          sandhiType = 'yi_sandhi';
          sandhiNote = 'sandhi';
        }
      }
      
      // Apply sandhi tolerance
      if (sandhiStatus) {
        modifiedResults[i].status = sandhiStatus;
        modifiedResults[i].sandhi = true;
        modifiedResults[i].sandhiType = sandhiType;
        modifiedResults[i].sandhiNote = sandhiNote;
        modifiedResults[i].originalPrediction = predictedTok;
      }
    }
    
    return modifiedResults;
  }

  return {
    stripTone,
    toneId,
    splitInitialFinal,
    isNgPair,
    isMinimalPair,
    toneClose,
    computeStatus,
    evaluateSyllable,
    applySandhiTolerance
  };
})();
