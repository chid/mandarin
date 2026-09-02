/**
 * Audio processing module for Mel spectrogram extraction.
 * Matches the Python implementation in ear/data/aishell.py using torchaudio
 */

const AudioProcessor = (function() {
  const SAMPLE_RATE = 16000;
  const N_FFT = 400;
  const HOP_LENGTH = 160;
  const N_MELS = 80;
  const F_MIN = 0;
  const F_MAX = SAMPLE_RATE / 2;

  // Precomputed Mel filterbank
  let melFilterbank = null;
  let hannWindow = null;

  /**
   * Initialize the Hann window and Mel filterbank
   */
  function init() {
    // Create Hann window (periodic, matching torch.hann_window)
    hannWindow = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
      hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
    }

    // Create Mel filterbank (matching torchaudio's implementation with Slaney normalization)
    melFilterbank = createMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS, F_MIN, F_MAX);
  }

  /**
   * Convert frequency to Mel scale (HTK formula, matching torchaudio default)
   */
  function hzToMel(hz) {
    return 2595.0 * Math.log10(1.0 + hz / 700.0);
  }

  /**
   * Convert Mel to frequency
   */
  function melToHz(mel) {
    return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
  }

  /**
   * Create Mel filterbank matrix matching torchaudio's melscale_fbanks
   * Uses Slaney normalization (area normalization)
   */
  function createMelFilterbank(sampleRate, nFft, nMels, fMin, fMax) {
    const nFreqs = Math.floor(nFft / 2) + 1;
    
    // Compute Mel points
    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);
    
    // nMels + 2 points for nMels triangular filters
    const melPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
      melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
    }

    // Convert Mel points to Hz
    const hzPoints = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
      hzPoints[i] = melToHz(melPoints[i]);
    }

    // Convert Hz to FFT bin (using linear interpolation like torchaudio)
    const freqBins = new Float32Array(nFreqs);
    for (let i = 0; i < nFreqs; i++) {
      freqBins[i] = i * sampleRate / nFft;
    }

    // Create filterbank
    const filterbank = [];
    for (let m = 0; m < nMels; m++) {
      const filter = new Float32Array(nFreqs);
      const fLeft = hzPoints[m];
      const fCenter = hzPoints[m + 1];
      const fRight = hzPoints[m + 2];

      for (let k = 0; k < nFreqs; k++) {
        const freq = freqBins[k];
        
        if (freq >= fLeft && freq <= fCenter) {
          filter[k] = (freq - fLeft) / (fCenter - fLeft);
        } else if (freq > fCenter && freq <= fRight) {
          filter[k] = (fRight - freq) / (fRight - fCenter);
        } else {
          filter[k] = 0;
        }
      }

      // Note: torchaudio uses norm=None by default, so no normalization
      // (Slaney normalization would be: enorm = 2.0 / (hzPoints[m + 2] - hzPoints[m]))
      
      filterbank.push(filter);
    }

    return filterbank;
  }

  /**
   * Optimized FFT using Cooley-Tukey algorithm
   */
  function fft(real, imag) {
    const n = real.length;
    if (n <= 1) return;

    // Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < n - 1; i++) {
      if (i < j) {
        [real[i], real[j]] = [real[j], real[i]];
        [imag[i], imag[j]] = [imag[j], imag[i]];
      }
      let k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey iterative FFT
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const angleStep = -2 * Math.PI / len;
      for (let i = 0; i < n; i += len) {
        let angle = 0;
        for (let k = 0; k < halfLen; k++) {
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const evenIdx = i + k;
          const oddIdx = i + k + halfLen;
          
          const tReal = real[oddIdx] * cos - imag[oddIdx] * sin;
          const tImag = real[oddIdx] * sin + imag[oddIdx] * cos;
          
          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] += tReal;
          imag[evenIdx] += tImag;
          
          angle += angleStep;
        }
      }
    }
  }

  /**
   * Compute power spectrum using DFT (direct computation for exact n_fft size)
   * Returns power spectrum (magnitude squared)
   * 
   * Note: Using DFT instead of FFT to match torchaudio's exact n_fft=400 behavior.
   * FFT would require padding to 512 which changes frequency resolution.
   */
  function powerSpectrum(frame) {
    const n = frame.length;
    const nFreqs = Math.floor(n / 2) + 1;
    const power = new Float32Array(nFreqs);

    // Direct DFT computation for first half of frequencies
    for (let k = 0; k < nFreqs; k++) {
      let real = 0;
      let imag = 0;
      const angle = -2 * Math.PI * k / n;
      for (let t = 0; t < n; t++) {
        real += frame[t] * Math.cos(angle * t);
        imag += frame[t] * Math.sin(angle * t);
      }
      power[k] = real * real + imag * imag;
    }
    return power;
  }

  /**
   * Compute STFT power spectrogram
   */
  function stft(audio) {
    // Pad audio at the beginning to center the first frame (matching torchaudio center=True default)
    const padAmount = Math.floor(N_FFT / 2);
    const paddedAudio = new Float32Array(audio.length + 2 * padAmount);
    // Reflect padding (matching torchaudio's default)
    for (let i = 0; i < padAmount; i++) {
      paddedAudio[padAmount - 1 - i] = audio[Math.min(i + 1, audio.length - 1)];
      paddedAudio[padAmount + audio.length + i] = audio[Math.max(audio.length - 2 - i, 0)];
    }
    for (let i = 0; i < audio.length; i++) {
      paddedAudio[padAmount + i] = audio[i];
    }

    const nFrames = Math.floor((paddedAudio.length - N_FFT) / HOP_LENGTH) + 1;
    if (nFrames <= 0) {
      return [];
    }

    const spectrogram = [];
    const frame = new Float32Array(N_FFT);

    for (let i = 0; i < nFrames; i++) {
      const start = i * HOP_LENGTH;
      
      // Apply window
      for (let j = 0; j < N_FFT; j++) {
        if (start + j < paddedAudio.length) {
          frame[j] = paddedAudio[start + j] * hannWindow[j];
        } else {
          frame[j] = 0;
        }
      }

      // Compute power spectrum
      const power = powerSpectrum(frame);
      spectrogram.push(power);
    }

    return spectrogram;
  }

  /**
   * Apply Mel filterbank to spectrogram
   */
  function applyMelFilterbank(spectrogram) {
    const melSpec = [];

    for (const frame of spectrogram) {
      const melFrame = new Float32Array(N_MELS);
      for (let m = 0; m < N_MELS; m++) {
        let sum = 0;
        for (let k = 0; k < frame.length; k++) {
          sum += frame[k] * melFilterbank[m][k];
        }
        // Clamp to avoid log of zero
        melFrame[m] = Math.max(sum, 1e-10);
      }
      melSpec.push(melFrame);
    }

    return melSpec;
  }

  /**
   * Convert power to dB scale (matching torchaudio.transforms.AmplitudeToDB with stype='power')
   * Formula: 10 * log10(max(x, amin))
   * Note: torchaudio uses top_db=None by default, so no dynamic range limiting
   */
  function powerToDb(melSpec) {
    const dbSpec = [];
    const amin = 1e-10;  // Minimum value to avoid log(0), matching torchaudio default

    for (const frame of melSpec) {
      const dbFrame = new Float32Array(N_MELS);
      for (let i = 0; i < N_MELS; i++) {
        // Power to dB: 10 * log10(max(x, amin))
        dbFrame[i] = 10.0 * Math.log10(Math.max(frame[i], amin));
      }
      dbSpec.push(dbFrame);
    }

    return dbSpec;
  }

  /**
   * Apply CMVN normalization (per-utterance, per-frequency)
   * Matches the Python: (feat - mean) / std
   */
  function applyCMVN(melSpec) {
    if (melSpec.length === 0) return melSpec;

    const nFrames = melSpec.length;
    const nMels = melSpec[0].length;

    // Compute mean per frequency bin
    const mean = new Float32Array(nMels);
    for (let m = 0; m < nMels; m++) {
      let sum = 0;
      for (let t = 0; t < nFrames; t++) {
        sum += melSpec[t][m];
      }
      mean[m] = sum / nFrames;
    }

    // Compute std per frequency bin (unbiased=False, matching PyTorch)
    const std = new Float32Array(nMels);
    for (let m = 0; m < nMels; m++) {
      let sqSum = 0;
      for (let t = 0; t < nFrames; t++) {
        const diff = melSpec[t][m] - mean[m];
        sqSum += diff * diff;
      }
      std[m] = Math.sqrt(sqSum / nFrames);
      // Clamp std to avoid division by zero
      if (std[m] < 1e-4) std[m] = 1e-4;
    }

    // Normalize
    const normalized = [];
    for (let t = 0; t < nFrames; t++) {
      const frame = new Float32Array(nMels);
      for (let m = 0; m < nMels; m++) {
        frame[m] = (melSpec[t][m] - mean[m]) / std[m];
      }
      normalized.push(frame);
    }

    return normalized;
  }

  /**
   * Resample audio to target sample rate using linear interpolation
   */
  function resample(audio, fromRate, toRate) {
    if (fromRate === toRate) return audio;

    const ratio = fromRate / toRate;
    const newLength = Math.floor(audio.length / ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIdx = i * ratio;
      const srcIdxFloor = Math.floor(srcIdx);
      const frac = srcIdx - srcIdxFloor;

      if (srcIdxFloor + 1 < audio.length) {
        resampled[i] = audio[srcIdxFloor] * (1 - frac) + audio[srcIdxFloor + 1] * frac;
      } else {
        resampled[i] = audio[srcIdxFloor];
      }
    }

    return resampled;
  }

  /**
   * Convert stereo to mono by averaging channels
   */
  function toMono(channelData) {
    if (channelData.length === 1) {
      return channelData[0];
    }

    const length = channelData[0].length;
    const mono = new Float32Array(length);
    const numChannels = channelData.length;

    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        sum += channelData[ch][i];
      }
      mono[i] = sum / numChannels;
    }

    return mono;
  }

  /**
   * Decode audio file to PCM samples
   */
  async function decodeAudio(audioBuffer) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: SAMPLE_RATE
    });

    try {
      const decoded = await audioContext.decodeAudioData(audioBuffer.slice(0));
      
      // Get all channels
      const channelData = [];
      for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
        channelData.push(decoded.getChannelData(ch));
      }

      // Convert to mono
      let audio = toMono(channelData);

      // Resample if needed (the AudioContext should handle this, but just in case)
      if (decoded.sampleRate !== SAMPLE_RATE) {
        audio = resample(audio, decoded.sampleRate, SAMPLE_RATE);
      }

      audioContext.close();
      return audio;
    } catch (e) {
      audioContext.close();
      throw e;
    }
  }

  /**
   * Process audio and return Mel spectrogram features
   * Returns { features: Float32Array (T x 80), length: number }
   */
  async function processAudio(audioData, maxSeconds = 5.0) {
    if (!melFilterbank) {
      init();
    }

    // Decode audio
    let audio;
    if (audioData instanceof ArrayBuffer) {
      audio = await decodeAudio(audioData);
    } else if (audioData instanceof Float32Array) {
      audio = audioData;
    } else {
      throw new Error('Invalid audio data type');
    }

    // Truncate to max length
    const maxSamples = Math.floor(maxSeconds * SAMPLE_RATE);
    if (audio.length > maxSamples) {
      audio = audio.slice(0, maxSamples);
    }

    // Compute spectrogram
    const spectrogram = stft(audio);
    
    if (spectrogram.length === 0) {
      return { features: new Float32Array(0), length: 0, hopLengthMs: HOP_LENGTH * 1000 / SAMPLE_RATE };
    }

    // Apply Mel filterbank
    const melSpec = applyMelFilterbank(spectrogram);
    
    // Convert to dB
    const dbSpec = powerToDb(melSpec);
    
    // Apply CMVN
    const normalized = applyCMVN(dbSpec);

    // Convert to flat Float32Array for ONNX (T x 80)
    const T = normalized.length;
    const features = new Float32Array(T * N_MELS);
    for (let t = 0; t < T; t++) {
      for (let m = 0; m < N_MELS; m++) {
        features[t * N_MELS + m] = normalized[t][m];
      }
    }

    return {
      features: features,
      length: T,
      hopLengthMs: HOP_LENGTH * 1000 / SAMPLE_RATE
    };
  }

  // Initialize on load
  init();

  return {
    processAudio,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_MELS
  };
})();
