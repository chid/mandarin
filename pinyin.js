/**
 * Pinyin conversion module.
 * Matches the Python implementation in ear/data/tokenizer.py
 */

const PinyinUtils = (function() {
  // Regex to strip non-pinyin characters (matches Python _TONE3_RE)
  const TONE3_RE = /[^a-z0-9:ü]+/gi;

  /**
   * Normalize pinyin token to match vocab format.
   * - Replace ü/u: with v
   * - Strip non-alphanumeric except tone digits
   * - Add tone 5 if no tone digit present
   */
  function normalizePinyinToken(tok) {
    if (!tok) return '';
    
    // Canonicalize ü/u: to v for ASCII consistency
    tok = tok.replace(/u:/g, 'v').replace(/ü/g, 'v');
    
    // Remove non-pinyin characters
    tok = tok.replace(TONE3_RE, '');
    
    // Convert to lowercase
    tok = tok.toLowerCase();
    
    // Add tone 5 (neutral tone) if no tone digit present
    // Also convert tone 0 to tone 5 (pinyin-pro uses 0 for neutral tone)
    if (tok && !/[0-9]$/.test(tok)) {
      tok = tok + '5';
    } else if (tok && tok.endsWith('0')) {
      tok = tok.slice(0, -1) + '5';
    }
    
    return tok;
  }

  /**
   * Convert Chinese text to pinyin tokens with tone numbers.
   * Uses pinyin-pro library (must be loaded via CDN).
   * Example: "中国" -> ["zhong1", "guo2"]
   */
  function textToPinyinTokens(text) {
    if (!text) return [];
    
    // Remove spaces
    text = text.replace(/\s+/g, '');
    if (!text) return [];

    // Check if pinyin-pro is available
    if (typeof pinyinPro === 'undefined') {
      console.error('pinyin-pro library not loaded');
      return [];
    }

    // Use pinyin-pro to get pinyin with tone numbers
    const result = pinyinPro.pinyin(text, {
      toneType: 'num',  // Use numeric tones (1-5)
      type: 'array',    // Return as array
      nonZh: 'removed'  // Remove non-Chinese characters
    });

    // Normalize each token
    const tokens = [];
    for (const item of result) {
      const normalized = normalizePinyinToken(item);
      if (normalized) {
        tokens.push(normalized);
      }
    }

    return tokens;
  }

  /**
   * Get characters and their pinyin tokens as pairs.
   * Returns array of [char, pinyinToken] tuples.
   * 
   * Uses full sentence context to correctly handle polyphonic characters (多音字)
   * like 了 (le vs liǎo), 得 (de vs dé vs děi), etc.
   */
  function charsAndPinyin(text) {
    if (!text) return [];
    
    // Remove spaces
    text = text.replace(/\s+/g, '');
    if (!text) return [];

    if (typeof pinyinPro === 'undefined') {
      console.error('pinyin-pro library not loaded');
      return [];
    }

    // Get pinyin for the ENTIRE sentence at once - this allows pinyin-pro
    // to use context for polyphonic characters (多音字)
    const pinyinResult = pinyinPro.pinyin(text, {
      toneType: 'num',
      type: 'array',
      nonZh: 'removed'
    });

    // Extract only Chinese characters from the text
    const chineseChars = [];
    for (const char of text) {
      if (/[\u4e00-\u9fff]/.test(char)) {
        chineseChars.push(char);
      }
    }

    // Pair characters with their pinyin
    const pairs = [];
    const minLength = Math.min(chineseChars.length, pinyinResult.length);
    
    for (let i = 0; i < minLength; i++) {
      const normalized = normalizePinyinToken(pinyinResult[i]);
      if (normalized) {
        pairs.push([chineseChars[i], normalized]);
      }
    }

    return pairs;
  }

  /**
   * Strip tone number from pinyin token.
   * Example: "zhong1" -> "zhong"
   */
  function stripTone(token) {
    if (!token) return token;
    if (/[0-9]$/.test(token)) {
      return token.slice(0, -1);
    }
    return token;
  }

  /**
   * Get tone ID (0-4) from pinyin token.
   * Tone 1-5 maps to 0-4.
   * Returns null if no valid tone.
   */
  function toneId(token) {
    if (!token) return null;
    const lastChar = token[token.length - 1];
    if (!/[1-5]/.test(lastChar)) return null;
    return parseInt(lastChar, 10);
  }

  return {
    normalizePinyinToken,
    textToPinyinTokens,
    charsAndPinyin,
    stripTone,
    toneId
  };
})();
