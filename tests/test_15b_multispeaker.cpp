// Multi-speaker closed-loop test for the 1.5B path.
//
// Synthesize a two-line dialog conditioned on two distinct reference
// WAVs (Speaker 0 + Speaker 1), then run ASR-7B on the result and
// confirm both speakers are diarized and each speaker's line has at
// least 60% word recall against the source.
//
// Skips with rc=77 unless these env vars all point at valid files:
//   VIBEVOICE_TTS_15B_MODEL  -> 1.5B gguf
//   VIBEVOICE_ASR_MODEL      -> ASR-7B gguf
//   VIBEVOICE_TOKENIZER      -> tokenizer gguf
//   VIBEVOICE_REF_WAV        -> Speaker 0's reference (24 kHz mono)
//   VIBEVOICE_REF_WAV_2      -> Speaker 1's reference (24 kHz mono)

#include "audio_io.hpp"
#include "vibevoice_asr.hpp"
#include "vibevoice_tts.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool file_ok(const char* p) {
    if (!p || !*p) return false;
    FILE* f = std::fopen(p, "rb");
    if (!f) return false;
    std::fclose(f); return true;
}

std::set<std::string> word_set(const std::string& s) {
    std::string clean;
    clean.reserve(s.size());
    for (char c : s) {
        clean += std::isalnum(static_cast<unsigned char>(c))
                 ? static_cast<char>(std::tolower(static_cast<unsigned char>(c)))
                 : ' ';
    }
    std::set<std::string> out;
    std::istringstream iss(clean);
    for (std::string w; iss >> w; ) out.insert(w);
    return out;
}

// Walk the ASR JSON output and return a vector of (speaker_id, content)
// pairs. Same dumb parser style as the single-speaker test — good
// enough for the model's stable JSON shape.
struct Utterance { int speaker = -1; std::string content; };

std::vector<Utterance> parse_utterances(const std::string& s) {
    std::vector<Utterance> out;
    size_t pos = 0;
    while (true) {
        const size_t spk = s.find("\"Speaker\":", pos);
        if (spk == std::string::npos) break;
        Utterance u;
        u.speaker = std::atoi(s.c_str() + spk + std::strlen("\"Speaker\":"));
        const std::string key = "\"Content\":\"";
        const size_t ck = s.find(key, spk);
        if (ck == std::string::npos) break;
        const size_t cs = ck + key.size();
        size_t ce = cs;
        while (ce < s.size() && s[ce] != '"') {
            ce += (s[ce] == '\\' && ce + 1 < s.size()) ? 2 : 1;
        }
        u.content = s.substr(cs, ce - cs);
        out.push_back(std::move(u));
        pos = ce;
    }
    return out;
}

double recall(const std::string& src, const std::string& got) {
    auto src_w = word_set(src);
    auto got_w = word_set(got);
    if (src_w.empty()) return 0.0;
    size_t hits = 0;
    for (const auto& w : src_w) if (got_w.count(w)) ++hits;
    return static_cast<double>(hits) / src_w.size();
}

}  // namespace

int main() {
    const char* tts15b = std::getenv("VIBEVOICE_TTS_15B_MODEL");
    const char* asr_p  = std::getenv("VIBEVOICE_ASR_MODEL");
    const char* tok_p  = std::getenv("VIBEVOICE_TOKENIZER");
    const char* ref_a  = std::getenv("VIBEVOICE_REF_WAV");
    const char* ref_b  = std::getenv("VIBEVOICE_REF_WAV_2");

    if (!file_ok(tts15b) || !file_ok(asr_p) || !file_ok(tok_p)
        || !file_ok(ref_a) || !file_ok(ref_b)) {
        std::fprintf(stderr,
            "skip: multi-speaker test needs VIBEVOICE_{TTS_15B_MODEL,"
            "ASR_MODEL,TOKENIZER,REF_WAV,REF_WAV_2} all set to valid "
            "files (got 15b=%s asr=%s tok=%s ref0=%s ref1=%s)\n",
            tts15b ? tts15b : "(null)", asr_p ? asr_p : "(null)",
            tok_p ? tok_p : "(null)", ref_a ? ref_a : "(null)",
            ref_b ? ref_b : "(null)");
        return 77;
    }

    // ---- 1. load 1.5B + tokenizer ----
    vv::VibeVoiceModel m15;
    if (!vv::vibevoice_load(tts15b, &m15)) { std::fprintf(stderr, "FAIL: load 1.5B\n"); return 1; }
    if (!m15.tokenizer.load_from_file(tok_p)) { std::fprintf(stderr, "FAIL: tokenizer load\n"); return 2; }
    if (m15.variant != "1.5b") {
        std::fprintf(stderr, "FAIL: variant=%s want 1.5b\n", m15.variant.c_str());
        return 3;
    }

    // ---- 2. synthesize the dialog ----
    // The 1.5B model uses the Text-input dialog as a "schedule" of
    // turns; with only two lines in the script it tends to emit
    // speech_end after speaker 0's turn. A third line gives it enough
    // runway to switch voices reliably (we still only check coverage
    // of the first two lines below).
    const std::string speaker0_line = "Hello there, this is the first speaker.";
    const std::string speaker1_line = "And this is the second speaker responding.";
    const std::string dialog =
        " Speaker 0: " + speaker0_line + "\n"
        " Speaker 1: " + speaker1_line + "\n"
        " Speaker 0: Nice to meet you.";

    vv::VibeVoiceTTSParams p;
    p.ref_audio_paths   = {ref_a, ref_b};
    p.max_speech_frames = 200;
    p.n_diffusion_steps = 20;
    p.cfg_scale         = 1.3f;
    p.seed              = 42;
    p.verbose           = false;

    std::vector<float> samples;
    if (vv::vibevoice_tts_generate(&m15, dialog, p, &samples) != 0
        || samples.empty()) {
        std::fprintf(stderr, "FAIL: tts produced no audio\n"); return 4;
    }
    std::printf("[multi] TTS produced %zu samples (%.2fs)\n",
                samples.size(), samples.size() / 24000.0);

    const std::string out_wav = "/tmp/vibevoice_15b_multi.wav";
    {
        vv_audio a{};
        a.samples     = samples.data();
        a.n_samples   = samples.size();
        a.sample_rate = 24000;
        a.channels    = 1;
        if (vv::save_wav_pcm16(out_wav.c_str(), a) != 0) {
            std::fprintf(stderr, "FAIL: save wav\n"); return 5;
        }
    }

    // ---- 3. transcribe + verify diarization + per-speaker recall ----
    vv::VibeVoiceModel masr;
    if (!vv::vibevoice_load(asr_p, &masr)) { std::fprintf(stderr, "FAIL: load ASR\n"); return 6; }
    if (!masr.tokenizer.load_from_file(tok_p)) { std::fprintf(stderr, "FAIL: ASR tokenizer\n"); return 7; }

    vv::VibeVoiceASRParams ap;
    ap.max_new_tokens     = 256;
    // Leave repetition_penalty / no_repeat_ngram at their struct
    // defaults (1.05f / 0) — matches the CLI, and the default
    // diarizes multi-speaker output more reliably than rep_pen=1.0.
    ap.verbose            = false;

    std::string transcript;
    if (vv::vibevoice_asr_transcribe(&masr, samples, ap, &transcript) != 0) {
        std::fprintf(stderr, "FAIL: ASR transcribe\n"); return 8;
    }
    std::printf("[multi] transcript: %s\n", transcript.c_str());

    auto utts = parse_utterances(transcript);
    if (utts.empty()) {
        std::fprintf(stderr, "FAIL: no utterances parsed from JSON\n"); return 9;
    }

    // Concatenate per-speaker content.
    std::string spk0_text, spk1_text;
    std::set<int> seen_speakers;
    for (const auto& u : utts) {
        seen_speakers.insert(u.speaker);
        if (u.speaker == 0) {
            if (!spk0_text.empty()) spk0_text += ' ';
            spk0_text += u.content;
        } else if (u.speaker == 1) {
            if (!spk1_text.empty()) spk1_text += ' ';
            spk1_text += u.content;
        }
    }

    if (seen_speakers.count(0) == 0 || seen_speakers.count(1) == 0) {
        std::fprintf(stderr,
            "FAIL: ASR diarized only %zu speaker(s); want both 0 and 1\n",
            seen_speakers.size());
        return 10;
    }

    const double r0 = recall(speaker0_line, spk0_text);
    const double r1 = recall(speaker1_line, spk1_text);
    std::printf("[multi] speaker 0 recall: %.1f%% (got: %s)\n", r0 * 100.0, spk0_text.c_str());
    std::printf("[multi] speaker 1 recall: %.1f%% (got: %s)\n", r1 * 100.0, spk1_text.c_str());

    // Looser threshold than single-speaker (0.80) — multi-speaker
    // alignment is harder, and our cfg=1.3 default trades a bit of
    // recall for stability.
    if (r0 < 0.6 || r1 < 0.6) {
        std::fprintf(stderr,
            "FAIL: per-speaker recall below 0.60 floor (s0=%.2f s1=%.2f)\n",
            r0, r1);
        return 11;
    }
    return 0;
}
