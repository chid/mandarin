"""
Ear Mandarin Pronunciation Evaluator (Python Implementation)
Matches the exact model and scoring algorithms described in:
https://simedw.com/2026/01/31/ear-pronunication-via-ctc/
"""

import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import onnxruntime as ort
import pypinyin
import soundfile as sf
from scipy.signal import resample_poly


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
F_MIN = 0.0
F_MAX = SAMPLE_RATE / 2.0

INITIALS = [
    'zh', 'ch', 'sh',  # Retroflexes first
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'j', 'q', 'x',
    'r', 'z', 'c', 's',
    'y', 'w'
]

MINIMAL_PAIR_INITIALS = {
    'zh-z', 'z-zh',
    'ch-c', 'c-ch',
    'sh-s', 's-sh',
    'j-q', 'q-j'
}


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sample_rate: int = SAMPLE_RATE,
                          n_fft: int = N_FFT,
                          n_mels: int = N_MELS,
                          f_min: float = F_MIN,
                          f_max: float = F_MAX) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    freq_bins = np.linspace(0, sample_rate / 2, n_freqs)
    
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]
        
        for k in range(n_freqs):
            freq = freq_bins[k]
            if f_left <= freq <= f_center:
                filterbank[m, k] = (freq - f_left) / (f_center - f_left)
            elif f_center < freq <= f_right:
                filterbank[m, k] = (f_right - freq) / (f_right - f_center)
                
    return filterbank


def compute_mel_spectrogram(audio: np.ndarray,
                            sample_rate: int = SAMPLE_RATE,
                            max_seconds: float = 5.0) -> Tuple[np.ndarray, int]:
    """
    Extracts 80-dim log Mel filterbank features with CMVN normalization,
    matching torchaudio preprocessing in the model.
    """
    max_samples = int(max_seconds * sample_rate)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    pad_amount = N_FFT // 2
    padded_audio = np.pad(audio, (pad_amount, pad_amount), mode='reflect')
    
    n_frames = (len(padded_audio) - N_FFT) // HOP_LENGTH + 1
    if n_frames <= 0:
        return np.empty((0, N_MELS), dtype=np.float32), 0

    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))
    frames = np.lib.stride_tricks.sliding_window_view(padded_audio, N_FFT)[::HOP_LENGTH]
    windowed_frames = frames * window
    
    stft_matrix = np.fft.rfft(windowed_frames, n=N_FFT, axis=1)
    power_spec = np.abs(stft_matrix) ** 2
    
    fb = create_mel_filterbank(sample_rate, N_FFT, N_MELS, F_MIN, F_MAX)
    mel_spec = np.dot(power_spec, fb.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    
    db_spec = 10.0 * np.log10(mel_spec)
    
    mean = np.mean(db_spec, axis=0, keepdims=True)
    std = np.std(db_spec, axis=0, keepdims=True)
    std = np.maximum(std, 1e-4)
    normalized = (db_spec - mean) / std
    
    return normalized.astype(np.float32), n_frames


def strip_tone(tok: str) -> str:
    if not tok:
        return tok
    if tok[-1].isdigit():
        return tok[:-1]
    return tok


def tone_id(tok: str) -> Optional[int]:
    if not tok:
        return None
    if tok[-1].isdigit() and tok[-1] in '12345':
        return int(tok[-1])
    return None


def split_initial_final(base: str) -> Tuple[str, str]:
    for ini in INITIALS:
        if base.startswith(ini):
            return ini, base[len(ini):]
    return '', base


def is_ng_pair(fin_a: str, fin_b: str) -> bool:
    pair = {fin_a, fin_b}
    return (
        pair == {'an', 'ang'} or
        pair == {'en', 'eng'} or
        pair == {'in', 'ing'} or
        pair == {'ian', 'iang'} or
        pair == {'uan', 'uang'}
    )


def is_minimal_pair(base_a: str, base_b: str) -> bool:
    ini_a, fin_a = split_initial_final(base_a)
    ini_b, fin_b = split_initial_final(base_b)

    if fin_a == fin_b and f"{ini_a}-{ini_b}" in MINIMAL_PAIR_INITIALS:
        return True
    if ini_a == ini_b and is_ng_pair(fin_a, fin_b):
        return True
    if (ini_a, ini_b) in [('l', 'n'), ('n', 'l'), ('f', 'h'), ('h', 'f')] and fin_a == fin_b:
        return True
    return False


def tone_close(tone_a: Optional[int], tone_b: Optional[int]) -> bool:
    if tone_a is None or tone_b is None:
        return False
    return {tone_a, tone_b} == {2, 3}


def compute_status(target_tok: str, top1_tok: str, p_target: float,
                   target_margin: float, top_confusion_tok: Optional[str]) -> Tuple[str, Optional[str]]:
    base_t = strip_tone(target_tok)
    base_p = strip_tone(top1_tok)
    tone_t = tone_id(target_tok)
    tone_p = tone_id(top1_tok)

    if p_target >= 0.85 and target_margin >= 0.5 and top1_tok == target_tok:
        return 'correct', None

    is_close = False
    if 0.5 <= p_target < 0.85:
        is_close = True
    if base_t == base_p and tone_close(tone_t, tone_p):
        is_close = True
    if top_confusion_tok and is_minimal_pair(base_t, strip_tone(top_confusion_tok)):
        is_close = True

    if is_close:
        if base_t == base_p and tone_t != tone_p:
            return 'close', 'tone'
        if base_t != base_p:
            ini_t, fin_t = split_initial_final(base_t)
            ini_p, fin_p = split_initial_final(base_p)
            if is_ng_pair(fin_t, fin_p) or ini_t != ini_p:
                return 'close', 'consonant'
            if fin_t != fin_p:
                return 'close', 'vowel'
        return 'close', None

    if base_t == base_p and tone_t != tone_p:
        return 'wrong', 'tone'

    ini_t, fin_t = split_initial_final(base_t)
    ini_p, fin_p = split_initial_final(base_p)
    if is_ng_pair(fin_t, fin_p) or ini_t != ini_p:
        return 'wrong', 'consonant'
    if fin_t != fin_p:
        return 'wrong', 'vowel'

    return 'wrong', None


def sentence_to_pinyin_tokens(text: str) -> List[Tuple[str, str]]:
    pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True)
    pairs = []
    clean_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
    
    char_idx = 0
    for p in pinyin_list:
        tok = p[0].lower().strip()
        if tok and not tok[-1].isdigit():
            tok += '5'
        if char_idx < len(clean_chars):
            pairs.append((clean_chars[char_idx], tok))
            char_idx += 1
            
    return pairs


class EarScorer:
    def __init__(self, model_path: str = "tiny_int8.onnx", vocab_path: str = "vocab.json"):
        base_dir = Path(__file__).resolve().parent
        m_path = Path(model_path)
        if not m_path.is_absolute():
            m_path = base_dir / model_path
        v_path = Path(vocab_path)
        if not v_path.is_absolute():
            v_path = base_dir / vocab_path
            
        with open(v_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.blank_id = self.token_to_id.get('<blank>', 0)
        self.unk_id = self.token_to_id.get('<unk>', 1)
        
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(m_path), session_opts, providers=['CPUExecutionProvider'])

    def log_softmax(self, logits: np.ndarray) -> np.ndarray:
        if logits.ndim == 3:
            logits = logits[0]
        max_val = np.max(logits, axis=-1, keepdims=True)
        exp_val = np.exp(logits - max_val)
        sum_exp = np.sum(exp_val, axis=-1, keepdims=True)
        return logits - max_val - np.log(sum_exp)

    def forced_align(self, log_probs: np.ndarray, target_ids: List[int]) -> np.ndarray:
        T, V = log_probs.shape
        n = len(target_ids)
        if n == 0:
            return np.full(T, -1, dtype=int)

        neg_inf = -1e10
        trellis = np.full((T + 1, n + 1), neg_inf, dtype=np.float32)
        trellis[0, 0] = 0.0

        for t in range(1, T + 1):
            lp = log_probs[t - 1]
            p_blank = lp[self.blank_id]

            trellis[t, 0] = trellis[t - 1, 0] + p_blank

            for j in range(1, n + 1):
                tok_id = target_ids[j - 1]
                p_tok = lp[tok_id]

                stay = trellis[t - 1, j] + p_blank
                adv = trellis[t - 1, j - 1] + p_tok

                trellis[t, j] = max(stay, adv)

        label_at_t = np.full(T, -1, dtype=int)
        curr_j = n

        for t in range(T, 0, -1):
            if curr_j == 0:
                break
            lp = log_probs[t - 1]
            tok_id = target_ids[curr_j - 1]

            stay = trellis[t - 1, curr_j] + lp[self.blank_id]
            adv = trellis[t - 1, curr_j - 1] + lp[tok_id]

            if adv >= stay and curr_j > 0:
                label_at_t[t - 1] = curr_j - 1
                curr_j -= 1

        return label_at_t

    def evaluate_syllable(self, log_probs: np.ndarray, target_tok: str, target_id: int, topk: int = 3) -> Dict[str, Any]:
        T, vocab_size = log_probs.shape
        if T == 0:
            return {
                'target': target_tok,
                'predicted': target_tok,
                'top1': target_tok,
                'status': 'wrong',
                'errorType': None,
                'pTarget': 0.0,
                'pTop1': 0.0,
                'pBlankAvg': 1.0,
                'top1Margin': 0.0,
                'targetMargin': 0.0,
                'topConfusions': []
            }

        probs = np.exp(log_probs)
        nonblank_mass = np.maximum(0.0, 1.0 - probs[:, 0])

        selected_frames = np.where(nonblank_mass >= 0.05)[0]
        if len(selected_frames) == 0:
            selected_frames = np.array([np.argmax(nonblank_mass)])

        used_t = len(selected_frames)
        p_blank_avg = float(np.mean(probs[selected_frames, 0]))

        # Conditioned nonblank probabilities
        avg_probs = np.zeros(vocab_size, dtype=np.float32)
        for t in selected_frames:
            denom = max(float(np.sum(probs[t, 1:])), 1e-8)
            avg_probs[1:] += probs[t, 1:] / denom
        avg_probs[1:] /= used_t

        # Exclude blank (idx 0)
        nonblank_scores = [(idx, float(avg_probs[idx])) for idx in range(1, vocab_size)]
        nonblank_scores.sort(key=lambda x: x[1], reverse=True)

        topk_ids = [s[0] for s in nonblank_scores[:topk]]
        top1_id = topk_ids[0] if topk_ids else target_id
        top1_tok = self.vocab[top1_id]

        p_target = float(avg_probs[target_id])
        p_top1 = float(avg_probs[top1_id])

        p_top2 = 0.0
        for idx in topk_ids:
            if idx != top1_id:
                p_top2 = float(avg_probs[idx])
                break
        top1_margin = p_top1 - p_top2

        best_alt = 0.0
        for idx in topk_ids:
            if idx != target_id:
                best_alt = float(avg_probs[idx])
                break
        target_margin = p_target - best_alt

        top_confusions = [self.vocab[idx] for idx in topk_ids if idx != target_id][:2]

        status, error_type = compute_status(
            target_tok,
            top1_tok,
            p_target,
            target_margin,
            top_confusions[0] if top_confusions else None
        )

        return {
            'target': target_tok,
            'predicted': top1_tok,
            'top1': top1_tok,
            'status': status,
            'errorType': error_type,
            'pTarget': round(p_target, 4),
            'pTop1': round(p_top1, 4),
            'pBlankAvg': round(p_blank_avg, 4),
            'top1Margin': round(top1_margin, 4),
            'targetMargin': round(target_margin, 4),
            'topConfusions': top_confusions
        }

    def apply_sandhi_tolerance(self, results: List[Dict[str, Any]], target_tokens: List[str]) -> List[Dict[str, Any]]:
        modified_results = [dict(r) for r in results]
        n = len(modified_results)

        for i in range(n):
            curr = modified_results[i]
            target_tok = target_tokens[i]
            predicted_tok = curr['top1']

            if curr['status'] == 'correct':
                continue

            target_base = strip_tone(target_tok)
            predicted_base = strip_tone(predicted_tok)
            target_t = tone_id(target_tok)
            predicted_t = tone_id(predicted_tok)

            if target_base != predicted_base:
                continue

            has_next = (i < n - 1)
            next_target_t = tone_id(target_tokens[i + 1]) if has_next else None

            sandhi_status = None
            sandhi_type = None
            sandhi_note = None

            # Rule 1: Third tone sandhi (3+3 context)
            if target_t == 3 and next_target_t == 3:
                if predicted_t == 2:
                    sandhi_status = 'correct'
                    sandhi_type = 'third_tone'
                    sandhi_note = 'sandhi'
                elif predicted_t == 3:
                    sandhi_status = 'correct'
                    sandhi_type = 'third_tone_careful'
                    sandhi_note = 'careful'

            # Rule 2: 不 (bu4) before tone 4 -> bu2
            if target_base == 'bu' and target_t == 4 and next_target_t == 4 and predicted_t == 2:
                sandhi_status = 'correct'
                sandhi_type = 'bu_sandhi'
                sandhi_note = 'sandhi'

            # Rule 3: 一 (yi1)
            if target_base == 'yi' and target_t == 1:
                if next_target_t == 4 and predicted_t == 2:
                    sandhi_status = 'correct'
                    sandhi_type = 'yi_sandhi'
                    sandhi_note = 'sandhi'
                elif next_target_t is not None and next_target_t in [1, 2, 3] and predicted_t == 4:
                    sandhi_status = 'correct'
                    sandhi_type = 'yi_sandhi'
                    sandhi_note = 'sandhi'

            if sandhi_status:
                curr['status'] = sandhi_status
                curr['sandhi'] = True
                curr['sandhiType'] = sandhi_type
                curr['sandhiNote'] = sandhi_note
                curr['originalPrediction'] = predicted_tok

        return modified_results

    def score_audio(self, audio_data: np.ndarray, sentence: str, sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
        if sample_rate != SAMPLE_RATE:
            gcd = math.gcd(sample_rate, SAMPLE_RATE)
            up = SAMPLE_RATE // gcd
            down = sample_rate // gcd
            audio_data = resample_poly(audio_data, up, down)

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=-1)

        feats, T = compute_mel_spectrogram(audio_data, SAMPLE_RATE, max_seconds=5.0)
        if T == 0:
            return {'error': 'Audio too short or empty', 'results': []}

        pairs = sentence_to_pinyin_tokens(sentence)
        if not pairs:
            return {'error': 'No Chinese characters found in sentence', 'results': []}

        chars = [p[0] for p in pairs]
        target_tokens = [p[1] for p in pairs]
        target_ids = [self.token_to_id.get(tok, self.unk_id) for tok in target_tokens]

        feats_tensor = np.expand_dims(feats, axis=0).astype(np.float32)
        feat_lens_tensor = np.array([T], dtype=np.int64)

        outputs = self.session.run(None, {'feats': feats_tensor, 'feat_lens': feat_lens_tensor})
        logits = outputs[0]
        out_T = logits.shape[1]

        log_probs = self.log_softmax(logits[0])
        label_at_t = self.forced_align(log_probs, target_ids)

        n_tokens = len(target_ids)
        token_ts = [[] for _ in range(n_tokens)]
        for t in range(out_T):
            lab = label_at_t[t]
            if 0 <= lab < n_tokens:
                token_ts[lab].append(t)

        results = []
        for idx in range(n_tokens):
            ts = token_ts[idx]
            if ts:
                center = (ts[0] + ts[-1]) // 2
                w = 3
                frame_indices = list(range(max(0, center - w), min(out_T, center + w + 1)))
            else:
                frame_indices = list(range(out_T))

            span_logp = log_probs[frame_indices]
            res = self.evaluate_syllable(span_logp, target_tokens[idx], target_ids[idx])
            res['char'] = chars[idx]
            results.append(res)

        results = self.apply_sandhi_tolerance(results, target_tokens)

        if n_tokens <= 2 and len(results) > 0:
            first = results[0]
            if first['status'] == 'wrong' and first['pTarget'] >= 0.15:
                confusions = first.get('topConfusions', [])
                has_close_confusion = any(
                    is_minimal_pair(strip_tone(first['target']), strip_tone(c)) or
                    tone_close(tone_id(first['target']), tone_id(c))
                    for c in confusions
                )
                if has_close_confusion or first['pTarget'] >= 0.3:
                    first['status'] = 'close'
                    first['shortUtteranceLeniency'] = True

        return {
            'sentence': sentence,
            'chars': chars,
            'targetTokens': target_tokens,
            'results': results
        }


def format_cli_output(evaluation: Dict[str, Any]) -> str:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    if 'error' in evaluation:
        return f"{RED}Error: {evaluation['error']}{RESET}"

    lines = []
    lines.append(f"\n{BOLD}Sentence:{RESET} {evaluation['sentence']}")
    lines.append("-" * 60)
    lines.append(f"{'Char':<6} {'Target':<10} {'Heard':<14} {'Target Prob':<14} {'Status'}")
    lines.append("-" * 60)

    for r in evaluation['results']:
        status = r['status']
        if status == 'correct':
            color = GREEN
            badge = "✓ CORRECT"
        elif status == 'close':
            color = YELLOW
            badge = "~ CLOSE"
        else:
            color = RED
            badge = "✗ WRONG"

        sandhi_info = " (sandhi)" if r.get('sandhi') else ""
        heard_text = r['top1'] + sandhi_info
        lines.append(
            f"{r['char']:<6} {r['target']:<10} {heard_text:<14} {r['pTarget']*100:>5.1f}%          {color}{badge}{RESET}"
        )

    lines.append("-" * 60)
    return "\n".join(lines)
