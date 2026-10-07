#include "phonon.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "third_party/json.hpp"

namespace phonon {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

template <typename T>
std::vector<T> read_raw(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t bytes = f.tellg();
  if (bytes != expected_count * sizeof(T))
    throw std::runtime_error(path + ": " + std::to_string(bytes) + " bytes, expected " +
                             std::to_string(expected_count * sizeof(T)));
  std::vector<T> v(expected_count);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}

std::string lib(const Options& o, const std::string& name) {
  return o.lib_dir.empty() ? name : o.lib_dir + "/" + name;
}

}  // namespace

Phonon::Phonon(const Options& o) : rng_(o.seed) {
  const std::string dir = o.bundle_dir + "/";
  std::ifstream mf(dir + "metadata.json");
  if (!mf) throw std::runtime_error("no metadata.json in " + o.bundle_dir);
  nlohmann::json meta = nlohmann::json::parse(mf);

  const auto& gen = meta.at("generation");
  sample_rate_ = gen.at("sample_rate");
  frame_rate_ = gen.at("frame_rate");
  cache_slots_ = gen.at("cache_slots");
  prefill_tokens_ = gen.at("prefill_tokens");
  max_tokens_ = gen.at("max_tokens_per_chunk");
  float temperature = o.temperature >= 0 ? o.temperature : float(gen.at("temperature"));
  noise_std_ = std::sqrt(temperature);
  eos_threshold_ = gen.at("eos_threshold");
  min_frames_before_eos_ = gen.at("min_frames_before_eos");
  const auto& layout = gen.at("layout");
  layers2_ = 2 * int(layout.at("layers"));
  heads_ = layout.at("heads");
  head_dim_ = layout.at("head_dim");

  const auto& bos = gen.at("bos");
  size_t bos_count = 1;
  for (int d : bos.at("shape")) bos_count *= d;
  bos_ = read_raw<f16>(dir + std::string(bos.at("file")), bos_count);

  for (const auto& v : meta.at("voices")) {
    Voice voice;
    voice.name = v.at("name");
    voice.length = v.at("length");
    voice.kv = read_raw<f16>(dir + std::string(v.at("file")), size_t(layers2_) * heads_ * voice.length * head_dim_);
    voices_.push_back(std::move(voice));
  }

  // ptts's text front end ships in the bundle, one build per platform; --lib-dir is the fallback.
  const auto& text = meta.at("text");
#if defined(__ANDROID__)
  const char* platform = "android-arm64";
#else
  const char* platform = "linux-arm64";
#endif
  const auto& lib_entry = text.contains("library") ? text.at("library") : nlohmann::json("libptts_text.so");
  std::string library = lib_entry.is_object() ? std::string(lib_entry.at(platform)) : std::string(lib_entry);
  std::ifstream in_bundle(dir + library);
  text_ = std::make_unique<PttsText>(in_bundle ? dir + library : lib(o, library),
                                     dir + std::string(text.at("tokenizer")), text.value("lang", "en"));

  const auto& rt = meta.at("runtime");
  std::vector<std::pair<std::string, std::string>> files;
  std::string backend_lib;
  if (o.backend == "htp") {
    backend_lib = "libQnnHtp.so";
    files.push_back({"", dir + std::string(rt.at("context_binaries").begin().value().at("file"))});
  } else if (o.backend == "cpu" || o.backend == "gpu") {
    backend_lib = o.backend == "cpu" ? "libQnnCpu.so" : "libQnnGpu.so";
    for (const auto& [graph, file] : rt.at("dlcs").items()) files.push_back({graph, dir + std::string(file)});
  } else {
    throw std::runtime_error("unknown backend " + o.backend + " (htp, cpu or gpu)");
  }
  model_ = std::make_unique<QnnModel>(lib(o, backend_lib), lib(o, "libQnnSystem.so"), files);
}

std::vector<std::string> Phonon::voices() const {
  std::vector<std::string> names;
  for (const auto& v : voices_) names.push_back(v.name);
  return names;
}

std::string Phonon::default_voice() const {
  for (const auto& v : voices_)
    if (v.name == "default") return v.name;
  if (voices_.empty()) throw std::runtime_error("the bundle has no voices");
  return voices_.front().name;
}

std::vector<float> Phonon::synthesize(const std::string& text, const std::string& voice_name,
                                         const std::function<void(const float*, size_t)>& on_audio,
                                         Stats* stats_out) {
  const Voice* voice = nullptr;
  for (const auto& v : voices_)
    if (v.name == voice_name) voice = &v;
  if (!voice) throw std::runtime_error("no voice " + voice_name);

  Stats stats;
  std::vector<float> out;
  auto t0 = Clock::now();
  model_->set_performance_mode(true);
  for (const auto& chunk : text_->split(text, max_tokens_, frame_rate_)) {
    generate_chunk(chunk, *voice, on_audio, out, stats, t0);
    stats.chunks++;
  }
  model_->set_performance_mode(false);
  stats.total_ms = ms_since(t0);
  if (stats_out) *stats_out = stats;
  return out;
}

void Phonon::generate_chunk(const Chunk& chunk, const Voice& voice,
                               const std::function<void(const float*, size_t)>& on_audio, std::vector<float>& out,
                               Stats& stats, Clock::time_point start) {
  const std::vector<int>& ids = chunk.ids;
  const int n = ids.size();
  if (n == 0) return;
  // prefill runs in pieces of prefill_tokens_, each needing that many free slots.
  const int pieces = (n + prefill_tokens_ - 1) / prefill_tokens_;
  if (voice.length + (pieces - 1) * prefill_tokens_ + prefill_tokens_ > cache_slots_)
    throw std::runtime_error("text chunk too long for the cache (" + std::to_string(n) + " tokens)");
  const int frames_after_eos = chunk.frames_after_eos;
  const int max_frames = std::min(chunk.frame_budget, cache_slots_ - voice.length - n);

  // The cache starts as the voice prompt, in slots 0..length-1. Each chunk starts
  // from the voice again, as ptts does.
  const size_t row = size_t(head_dim_);
  const size_t plane = size_t(cache_slots_) * row;  // one (layer, k/v, head)
  std::vector<f16> kv(size_t(layers2_) * heads_ * plane, f16(0));
  for (int lh = 0; lh < layers2_ * heads_; lh++)
    std::memcpy(&kv[lh * plane], &voice.kv[size_t(lh) * voice.length * row], voice.length * row * sizeof(f16));
  int pos = voice.length;

  // Writes a [2L, H, s, D] block's first `count` positions into the cache at `at`.
  auto write_kv = [&](const std::vector<f16>& block, int s, int count, int at) {
    for (int lh = 0; lh < layers2_ * heads_; lh++)
      std::memcpy(&kv[lh * plane + size_t(at) * row], &block[size_t(lh) * s * row], count * row * sizeof(f16));
  };

  const Graph& prefill = model_->graph("prefill");
  const Graph& step = model_->graph("step");

  {
    auto t = Clock::now();
    std::vector<f16> kv_new(prefill.output("output_0").count);
    for (int first = 0; first < n; first += prefill_tokens_) {
      const int count = std::min(prefill_tokens_, n - first);
      std::vector<int32_t> tokens(prefill_tokens_, 0);
      std::copy(ids.begin() + first, ids.begin() + first + count, tokens.begin());
      int32_t pos32 = pos;
      model_->execute(prefill, {{"tokens", tokens.data()}, {"kv_cache", kv.data()}, {"pos", &pos32}},
                      {{"output_0", kv_new.data()}});
      write_kv(kv_new, prefill_tokens_, count, pos);
      pos += count;
    }
    stats.prefill_ms += ms_since(t);
  }

  // Step's buffers. Mimi states go in as mimi_states_i and come back as output_(5 + i).
  std::vector<f16> emb = bos_;
  std::vector<f16> noise(step.input("noise").count);
  std::vector<f16> audio(step.output("output_0").count);
  f16 eos = 0;
  std::vector<f16> next_emb(emb.size());
  std::vector<f16> kv_new(step.output("output_3").count);
  std::vector<f16> latent(step.output("output_4").count);
  std::vector<std::vector<f16>> states, next_states;
  for (int i = 0;; i++) {
    std::string name = "mimi_states_" + std::to_string(i);
    bool found = false;
    for (const auto& t : step.inputs) found |= t.name == name;
    if (!found) break;
    states.emplace_back(step.input(name).count, f16(0));
  }
  next_states = states;
  std::normal_distribution<float> normal(0.0f, noise_std_);

  int eos_frame = -1;
  for (int frame = 0; frame < max_frames; frame++) {
    auto t = Clock::now();
    for (auto& x : noise) x = f16(normal(rng_));
    int32_t pos32 = pos, frame32 = frame;
    std::map<std::string, const void*> in = {{"emb", emb.data()},   {"noise", noise.data()},  {"kv_cache", kv.data()},
                                             {"pos", &pos32},       {"frame", &frame32}};
    std::map<std::string, void*> outs = {{"output_0", audio.data()}, {"output_1", &eos},
                                         {"output_2", next_emb.data()}, {"output_3", kv_new.data()},
                                         {"output_4", latent.data()}};
    for (size_t i = 0; i < states.size(); i++) {
      in["mimi_states_" + std::to_string(i)] = states[i].data();
      outs["output_" + std::to_string(5 + i)] = next_states[i].data();
    }
    model_->execute(step, in, outs);
    stats.step_ms += ms_since(t);
    write_kv(kv_new, 1, 1, pos);
    pos++;
    emb.swap(next_emb);
    states.swap(next_states);

    if (eos_frame < 0 && frame >= min_frames_before_eos_ && float(eos) > eos_threshold_) eos_frame = frame;
    if (eos_frame >= 0 && frame >= eos_frame + frames_after_eos) break;

    size_t first = out.size();
    for (f16 s : audio) out.push_back(float(s));
    stats.frames++;
    if (stats.first_audio_ms < 0) stats.first_audio_ms = ms_since(start);
    if (on_audio) on_audio(out.data() + first, audio.size());
  }
}

}  // namespace phonon
