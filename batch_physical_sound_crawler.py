# -*- coding: utf-8 -*-
"""
batch_physical_sound_crawler.py: Large-scale Physical Sound Internet Crawler & Analyzer.

Processes onomatopoeia dictionary (JP: 2,061, KR: 5,050 entries):
  1. Filters sound/physical onomatopoeia (Phenomime/Phonomime/Sound keywords).
  2. Generates search keywords from English meanings and phonetic rules.
  3. Searches & downloads real sound effect audio from Freesound (via direct web preview scraping or API).
  4. Automatically converts to WAV via ffmpeg.
  5. Performs physical acoustic analysis (Low-frequency transient, Crest factor, Flatness, Decay).
  6. Maps features to DSP synth parameters and OnomaDict 16D vectors.
  7. Saves presets and audio with resume capability (skips already processed words).
"""

import sys
import os
import re
import json
import time
import argparse
import urllib.request
import urllib.parse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZER_SRC = PROJECT_ROOT / "onomato-audio-analyzer" / "src"
if str(ANALYZER_SRC) not in sys.path:
    sys.path.insert(0, str(ANALYZER_SRC))

from physical_audio_analyzer import PhysicalAudioAnalyzer, PhysicalAudioFeatures

# Default Output Directories
OUTPUT_DIR = PROJECT_ROOT / "dataset_output"
PRESET_DIR = OUTPUT_DIR / "physical_sound_presets"
WAV_DIR = OUTPUT_DIR / "physical_sound_wavs"
INDEX_FILE = OUTPUT_DIR / "physical_sound_index.json"

# Sound keywords for filtering
SOUND_KEYWORDS = [
    "sound of", "noise of", "clatter", "bang", "crash", "thud", "splash", "click", "rattle", "boom",
    "roar", "rustle", "buzz", "hiss", "creak", "crack", "drip", "crunch", "pop", "knock", "ring",
    "chime", "whistle", "whoosh", "screech", "snap", "tap", "chirp", "grunt", "groan", "hum",
    "sizzle", "gurgle", "splash", "slosh", "smack", "thump", "tick", "beep", "howl", "growl",
    "bark", "meow", "clack", "clang", "clink", "flap", "flutter", "murmur", "squish", "screaming",
    "explosion", "burst", "dripping", "flowing", "wind blowing", "strike", "hit", "drop", "friction",
    "shaking", "flutter", "rumble", "creaking", "scraping", "sliding", "rolling", "pounding"
]


# Semantic keyword dictionary for translating Japanese acoustic descriptors to rich Freesound queries
SEMANTIC_KEYWORD_MAP = {
    # Longest / compound phrases first
    "手拍子": "clapping applause hands",
    "拍手": "clapping applause hands",
    "衝撃波": "shockwave explosion blast",
    "爆発音": "explosion blast boom",
    "爆発": "explosion blast boom",
    "金属音": "metal impact clang",
    "金属打撃": "metal clang impact",
    "金属": "metal impact clang",
    "雨音": "rain rainfall drizzle",
    "水音": "water splash dripping",
    "流水": "running water stream",
    "流れる": "water flow stream",
    "流動": "flow stream liquid",
    "波の音": "ocean waves surf",
    "海の波": "ocean waves surf",
    "風の音": "wind breeze whoosh",
    "たき火": "campfire crackle fire",
    "火花": "spark crackle fire",
    "発火": "fire crackle spark",
    "足音": "footstep walking footsteps",
    "靴音": "shoes footsteps leather",
    "ベル": "bell ring chime",
    "鐘": "bell chime tolling",
    "鍵": "keys metal jingle",
    "鈴": "bell jingle chime",
    "木製": "wood wooden knock clack",
    "剛体": "wooden rattle clatter",
    "木": "wood wooden knock clack",
    "ガラス": "glass shatter break",
    "打撃": "heavy impact punch hit",
    "衝突": "crash collision impact",
    "銃声": "gunshot blast explosion",
    "破裂": "pop burst balloon",
    "弾ける": "pop snap spark",
    "軋み": "creak door wood squeak",
    "摩擦音": "friction scraping sliding",
    "摩擦": "friction scraping sliding",
    "擦る": "rubbing friction rustle",
    "衣擦れ": "cloth rustle fabric",
    "砂": "sand rustle granular",
    "粘性": "slime squish viscous",
    "液体": "liquid water squish",
    "泥": "mud squish slime",
    "乱流": "wind turbulence whoosh",
    "乾燥": "rustle dry grass",
    "振動": "vibration rumble hum",
    "笑い声": "laughter chuckle laughing",
    "笑い": "laughter chuckle laughing",
    "叫び声": "screaming shout yell",
    "叫び": "screaming shout yell",
    "泣き声": "crying weeping sob",
    "泣き": "crying weeping sob",
    "呻き": "groan grunt moan",
    "ため息": "sigh sighing breath",
    "咳": "cough coughing throat",
    "咆哮": "roar growl beast",
    "鳥の鳴き声": "bird chirp singing",
    "鳥": "bird chirp singing",
    "犬の鳴き声": "dog bark barking",
    "犬": "dog bark barking",
    "猫の鳴き声": "cat meow purr",
    "猫": "cat meow purr",
    "羽ばたき": "wings flapping bird",
    "心臓": "heartbeat pulse thud",
    "鼓動": "heartbeat thumping pulse",
    "雷": "thunder rumble lightning",
    "嵐": "storm wind thunder",
    "炎": "flame fire whoosh",
    "風": "wind breeze whoosh",
}


class QueryBuilder:
    """Generates clean, targeted Freesound search queries from onomatopoeia metadata."""

    CLEAN_RE = re.compile(r"[^a-zA-Z0-9\s]")
    STOPWORDS = {
        "manner", "or", "sound", "of", "state", "being", "action", "repeatedly", "continuous",
        "making", "a", "an", "the", "in", "with", "and", "by", "to", "for", "from", "on", "at",
        "like", "as", "something", "someone", "small", "large", "heavy", "light", "very", "loud",
        "condition", "phenomenon", "time", "decay", "hardness", "space", "flow", "weight"
    }

    PREFIX_RE = re.compile(r"^manner or sound of\s+[^:]+:\s*", re.IGNORECASE)

    @classmethod
    def build_query(cls, entry: Dict[str, Any]) -> str:
        word = entry.get("word", "")
        meaning_en = entry.get("meaning_en", "") or ""
        rationale = entry.get("rationale", "") or ""

        # 1. Semantic keyword match from Japanese description (longest match first)
        # e.g. "ぱちぱち" -> "拍手 or 発火" -> "clapping applause hands"
        # e.g. "どかん"   -> "爆発。最大の衝撃波..." -> "shockwave explosion blast" / "explosion blast boom"
        full_text = f"{meaning_en} {rationale}"
        for jp_key in sorted(SEMANTIC_KEYWORD_MAP.keys(), key=len, reverse=True):
            if jp_key in full_text:
                return SEMANTIC_KEYWORD_MAP[jp_key]

        # Strip artificial prefix "manner or sound of <word>:"
        body = cls.PREFIX_RE.sub("", meaning_en).strip()

        # 2. Extract keywords from English patterns: e.g. "sound of water splashing"
        m = re.search(r"(?:sound|noise) of ([\w\s]+?)(?:\.|\;|\,|$)", body, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            words = [w for w in cls.CLEAN_RE.sub(" ", candidate).split() if w.lower() not in cls.STOPWORDS]
            if words:
                return " ".join(words[:2])

        # 3. Fallback: extract important sound-related words from body
        body_lower = body.lower()
        found = [w for w in SOUND_KEYWORDS if re.search(rf"\b{re.escape(w)}\b", body_lower) and w not in ["sound of", "noise of", "sound", "noise"]]
        if found:
            return " ".join(found[:2])

        # 5. Phonetic acoustic feature derivation based on Japanese/Korean mora structure
        # (Never query raw non-ASCII word, translate phonemes to true acoustic texture)
        w_lower = word.lower()
        if any(c in w_lower for c in "かきくけこカキクケコk"):
            return "wood knock click tapping"
        elif any(c in w_lower for c in "ぱぴぷぺぽパピプペポp"):
            return "pop click snap small"
        elif any(c in w_lower for c in "さしすせそサシスセソs"):
            return "rustle whisper friction soft"
        elif any(c in w_lower for c in "ざじずぜぞザジズゼゾz"):
            return "heavy buzz vibration stream"
        elif any(c in w_lower for c in "ばびぶべぼバビブベボb"):
            return "heavy thud splash burst"
        elif any(c in w_lower for c in "だぢづでどダヂヅデドd"):
            return "heavy thud slam impact"
        elif any(c in w_lower for c in "がぎぐげごガギグゲゴg"):
            return "wooden rattle clatter rumble"
        elif any(c in w_lower for c in "たちつてとタチツテトt"):
            return "tap footstep click sharp"
        elif any(c in w_lower for c in "らりるれろラリルレロr"):
            return "rolling flow water stream"
        elif any(c in w_lower for c in "はひふへほハヒフヘホh"):
            return "wind breath gentle whoosh"
        elif any(c in w_lower for c in "まみむめもマミムメモm"):
            return "hum murmur gentle drone"

        return "acoustic percussion resonant tone"


class WebSoundDownloader:
    """Downloads sound preview files directly from Freesound via Web search or API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FREESOUND_API_KEY", "").strip()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def search_and_download(self, query: str, target_wav: Path) -> Tuple[bool, str, str]:
        """
        Searches Freesound and downloads the best audio preview, converted to mono WAV.
        Returns (success: bool, source_description: str, message: str).
        """
        # If API key is provided, use official API
        if self.api_key:
            ok, msg = self._download_via_api(query, target_wav)
            if ok:
                return True, "Freesound API (Official)", msg

        # Otherwise, use direct web search scraper
        ok, msg = self._download_via_web(query, target_wav)
        if ok:
            return True, "Freesound Web Search (Direct Preview)", msg

        return False, "None", msg

    def _download_via_web(self, query: str, target_wav: Path) -> Tuple[bool, str]:
        """Scrapes Freesound search page and downloads preview MP3 with multi-tier fallback."""
        words = query.split()
        candidate_queries = [query]
        if len(words) > 2:
            candidate_queries.append(" ".join(words[:2]))
        if len(words) >= 1:
            candidate_queries.append(f"{words[0]} sound")
        # STRICT RULE: NEVER fall back to unrelated 'foley impact' (rock/explosion shatter)!
        # If true matching sound doesn't exist, report failure cleanly.

        previews = []
        last_err = ""
        for q in candidate_queries:
            search_url = f"https://freesound.org/search/?q={urllib.parse.quote(q)}"
            req = urllib.request.Request(search_url, headers=self.headers)
            try:
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                previews = re.findall(r"https://cdn\.freesound\.org/previews/[^\s\"\'\<\>]+\.mp3", html)
                if not previews:
                    previews = re.findall(r"https://cdn\.freesound\.org/previews/[^\s\"\'\<\>]+\.ogg", html)
                if previews:
                    break
            except Exception as e:
                last_err = str(e)
                continue

        if not previews:
            return False, f"No preview audio found (tried: {candidate_queries}, err: {last_err})"

        # Prefer HQ preview if available, else first preview
        preview_url = previews[0]
        for p in previews:
            if "-hq." in p:
                preview_url = p
                break

        # Download preview to temp file
        temp_audio = target_wav.with_suffix(".temp_audio")
        try:
            dl_req = urllib.request.Request(preview_url, headers=self.headers)
            with urllib.request.urlopen(dl_req, timeout=15.0) as dl_resp:
                data = dl_resp.read()
                if len(data) < 1000:
                    return False, f"Downloaded audio too small ({len(data)} bytes)"
                with open(temp_audio, "wb") as f:
                    f.write(data)

            # Convert to standard WAV (22050Hz Mono) via ffmpeg
            cmd = [
                "ffmpeg", "-i", str(temp_audio),
                "-ar", "22050", "-ac", "1",
                "-t", "4.0",  # limit to max 4 seconds for sound analysis
                str(target_wav), "-y", "-loglevel", "error"
            ]
            subprocess.run(cmd, check=True)
            return True, f"Downloaded {len(data)} bytes from {preview_url}"
        except Exception as e:
            return False, f"Download/Conversion failed: {e}"
        finally:
            if temp_audio.exists():
                try:
                    temp_audio.unlink()
                except Exception:
                    pass

    def _download_via_api(self, query: str, target_wav: Path) -> Tuple[bool, str]:
        """Queries Freesound API v2 and downloads preview."""
        base_url = "https://freesound.org/apiv2/search/text/"
        params = {
            "query": query,
            "filter": "duration:[0.1 TO 4.0]",
            "fields": "id,name,previews,duration",
            "token": self.api_key,
            "page_size": 3,
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "OnomaPhysicalCrawler/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return False, f"No API results for '{query}'"

            first = results[0]
            previews = first.get("previews", {})
            preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
            if not preview_url:
                return False, "No preview link in API result"

            temp_audio = target_wav.with_suffix(".temp_audio")
            urllib.request.urlretrieve(preview_url, str(temp_audio))

            cmd = [
                "ffmpeg", "-i", str(temp_audio),
                "-ar", "22050", "-ac", "1",
                "-t", "4.0",
                str(target_wav), "-y", "-loglevel", "error"
            ]
            subprocess.run(cmd, check=True)
            return True, f"API sound ID: {first.get('id')} ({first.get('name')})"
        except Exception as e:
            return False, f"API error: {e}"
        finally:
            if temp_audio.exists():
                try:
                    temp_audio.unlink()
                except Exception:
                    pass


# Strict regex patterns for filtering Japanese onomatopoeia
JP_SOUND_RE = re.compile(
    r"\b(?:sound|noise|clatter|bang|crash|thud|splash|click|rattle|boom|rustle|buzz|hiss|creak|crack|drip|crunch|pop|knock|chime|whistle|whoosh|screech|snap|tap|sizzle|gurgle|slosh|smack|thump|tick|beep|clack|clang|clink|flap|squish|explosion|blast|rumble|dripping|ring|ringing|echo|reverberation|friction|vibration|impact|hit|strike|collision|shatter|fall|drop|pour|drizzle|flow|blow|stream|roar|cry|groan|grunt|howl|whimper|bark|screaming|voice|laughter|gasping|weeping|sighing|babbling|snicker|humming)\b"
    r"|(?:音|声|鳴き声|叫び|爆発|打撃|衝突|破裂|激突|きしみ|轟音|足音|水音|雨音|風の音|摩擦音|金属音|打音|衝撃音|振動|響き|反響|破砕|落下|流水|滴り|噴出|擦れ|擦る|叩く|殴る|砕ける|弾ける|鳴る|軋む|唸る|響く|揺れる|震える|拍手|手拍子|発火|火花|たき火|ベル|鐘)",
    re.IGNORECASE
)

JP_MIMETIC_EXCLUDE_RE = re.compile(
    r"^state of\b|^State of\b|^feeling of\b|^appearance of\b|^condition of\b|state/manner"
    r"|(?:油断|心が弾む|決断力|群がって|したいこと|過ち|かすかに|不快な|労働|狼狽|慌てる|のんびり|すっきり|さっぱり|うんざり|ぼんやり|うっとり|ぐっすり|そわそわ|どきどき|わくわく|はらはら|ひやひや|いらいら|うきうき|うかうか|うじうじ|うずうず|おどおど|おろおろ|くよくよ|ぐずぐず|じろじろ|きょろきょろ|うろうろ|ぶらぶら|ふらふら|よちよち|よろよろ|ふらり|ぐったり|ぐんぐん|すくすく|めきめき|どんどん|ずんずん|すらすら|するする|てきぱき|きびきび|のろのろ|もたもた|だらだら|たらたら|ぐだぐだ|だらり|ぽかん|あっけ|あっさり|あたふた|あんぐり|いじいじ|いそいそ|いちゃいちゃ)",
    re.IGNORECASE
)


def filter_sound_entries(dictionary: List[Dict[str, Any]], lang: str = "ALL") -> List[Dict[str, Any]]:
    """
    Filters target onomatopoeia representing strictly physical sounds and voices.
    Strictly excludes pure mimetic / psychological / manner words (擬態語・擬情語).
    """
    sound_cats = {"Phenomime (Sound)", "Phonomime (Voice)", "Sound"}
    candidates = []

    for entry in dictionary:
        l = entry.get("language", "")
        if lang != "ALL" and l != lang:
            continue

        cat = entry.get("category", "")
        meaning_en = entry.get("meaning_en") or ""
        rationale = entry.get("rationale") or ""

        is_sound = False

        if l in ["KR", "AF"]:
            # In Korean and Afrikaans, categories are explicitly tagged
            if cat in sound_cats:
                is_sound = True
        elif l == "JP":
            # For Japanese: strip template prefix and check true acoustic vs mimetic semantics
            body = QueryBuilder.PREFIX_RE.sub("", meaning_en).strip()
            # If explicit semantic key exists (e.g. 拍手, 爆発, 鐘) -> sound
            if any(k in f"{meaning_en} {rationale}" for k in SEMANTIC_KEYWORD_MAP):
                is_sound = True
            elif JP_MIMETIC_EXCLUDE_RE.search(body):
                # Pure mimetic (state, feeling, attitude) -> strictly exclude!
                is_sound = False
            elif JP_SOUND_RE.search(body):
                # Contains acoustic/physics words -> include
                is_sound = True

        if is_sound:
            candidates.append(entry)

    return candidates


def run_batch_crawler(
    lang: str = "ALL",
    limit: int = 0,
    delay: float = 0.8,
    api_key: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Executes the batch crawler across the onomatopoeia dictionary.
    """
    dict_file = PROJECT_ROOT / "data" / "onomatopoeia_dictionary_enriched.json"
    if not dict_file.exists():
        print(f"[Error] Dictionary file not found at {dict_file}")
        return

    print("=" * 70)
    print("  🌐 OnomaDict 大規模物理音インターネット・クローラー ＆ 音響解析エンジン")
    print("=" * 70)
    print(f"辞書ファイル: {dict_file}")
    print(f"対象言語: {lang} | 件数制限: {'無制限 (全件)' if limit <= 0 else f'{limit} 件'}")
    print(f"保存先プリセット: {PRESET_DIR}")
    print(f"保存先WAV: {WAV_DIR}")
    print("-" * 70)

    # 1. Load dictionary
    with open(dict_file, "r", encoding="utf-8") as f:
        full_dict = json.load(f)

    # 2. Filter sound candidates
    candidates = filter_sound_entries(full_dict, lang=lang)
    total_candidates = len(candidates)
    print(f"✔ 辞書全体 {len(full_dict):,} 語中、擬音語・擬声語の対象語: {total_candidates:,} 語 を抽出しました。")

    if limit > 0:
        candidates = candidates[:limit]
        print(f"※ 今回の実行件数: {len(candidates)} 語 に制限します。\n")
    else:
        print(f"※ 全 {total_candidates:,} 語の処理を開始します。\n")

    # Ensure output directories exist
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing index for resume
    index_data = {}
    if INDEX_FILE.exists():
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception:
            index_data = {}

    downloader = WebSoundDownloader(api_key=api_key)
    analyzer = PhysicalAudioAnalyzer(sample_rate=22050)

    success_count = 0
    skip_count = 0
    fail_count = 0
    start_time = time.time()

    print(f"{'No.':<6} | {'言語/単語':<16} | {'検索クエリ':<22} | {'状態・解析結果'}")
    print("-" * 70)

    for idx, entry in enumerate(candidates, 1):
        word = entry.get("word", "").strip()
        l_code = entry.get("language", "XX")
        safe_word = re.sub(r"[^\w\-]", "_", word)
        file_key = f"{l_code}_{safe_word}"

        preset_file = PRESET_DIR / f"{file_key}.json"
        target_wav = WAV_DIR / f"{file_key}.wav"

        # Check for resume
        if not overwrite and preset_file.exists() and target_wav.exists():
            skip_count += 1
            if idx % 50 == 0 or idx <= 5:
                print(f"[{idx:>4}/{len(candidates)}] {l_code}: {word:<12} | [スキップ] すでに解析済みです")
            continue

        # Build search query
        query = QueryBuilder.build_query(entry)

        # Download sound from internet
        print(f"[{idx:>4}/{len(candidates)}] {l_code}: {word:<12} | '{query}' 🌐 検索中...", end="\r", flush=True)

        ok, source, msg = downloader.search_and_download(query, target_wav)

        if not ok:
            fail_count += 1
            print(f"[{idx:>4}/{len(candidates)}] {l_code}: {word:<12} | '{query:<20}' | ❌ 該当音源なし: {msg[:30]}")
            continue

        # Perform Physical Audio Analysis
        try:
            audio, sr = analyzer.load_audio(target_wav)
            features = analyzer.extract_features(audio, sr)
            dsp_params = analyzer.generate_synth_parameters(features)
            onoma_vec = analyzer.map_to_onomadict_vector(features)

            # Build record
            record = {
                "word": word,
                "language": l_code,
                "category": entry.get("category"),
                "meaning_en": entry.get("meaning_en"),
                "search_query": query,
                "audio_source": source,
                "wav_file": str(target_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "physical_features": {
                    "duration_sec": round(features.duration_sec, 3),
                    "transient_low_freq_ratio": round(features.transient_low_freq_ratio, 4),
                    "crest_factor_db": round(features.crest_factor_db, 2),
                    "attack_time_ms": round(features.attack_time_ms, 2),
                    "spectral_flatness": round(features.spectral_flatness, 4),
                    "spectral_centroid_hz": round(features.spectral_centroid_hz, 1),
                    "decay_time_ms": round(features.decay_time_ms, 1),
                },
                "dsp_synth_parameters": dsp_params,
                "onomadict_16d_vector": onoma_vec,
            }

            # Save preset JSON
            with open(preset_file, "w", encoding="utf-8") as pf:
                json.dump(record, pf, ensure_ascii=False, indent=2)

            # Update index
            index_data[file_key] = {
                "word": word,
                "language": l_code,
                "category": entry.get("category"),
                "query": query,
                "sub_kick": dsp_params.get("sub_kick_on", False),
                "routing": dsp_params.get("routing_pattern", "D"),
                "material": dsp_params.get("modal_material", "friction"),
                "preset_path": str(preset_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "wav_path": str(target_wav.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }

            success_count += 1
            kick_str = "Sub-Kick:ON" if dsp_params.get("sub_kick_on") else "Sub-Kick:OFF"
            mat_str = dsp_params.get("modal_material", "")
            pattern_str = f"Pattern {dsp_params.get('routing_pattern', '')}"

            print(f"[{idx:>4}/{len(candidates)}] {l_code}: {word:<12} | '{query:<20}' | ✔ 完了 [{kick_str}, {mat_str}, {pattern_str}]")

        except Exception as e:
            fail_count += 1
            print(f"[{idx:>4}/{len(candidates)}] {l_code}: {word:<12} | '{query:<20}' | ❌ 解析エラー: {e}")

        # Periodically save index
        if success_count % 10 == 0:
            with open(INDEX_FILE, "w", encoding="utf-8") as idxf:
                json.dump(index_data, idxf, ensure_ascii=False, indent=2)

        # Politeness delay between internet requests
        time.sleep(delay)

    # Final index save
    with open(INDEX_FILE, "w", encoding="utf-8") as idxf:
        json.dump(index_data, idxf, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("=" * 70)
    print("【処理完了サマリー】")
    print(f"  - 処理完了（新規取得＆解析成功）: {success_count} 件")
    print(f"  - スキップ（既存）: {skip_count} 件")
    print(f"  - 失敗: {fail_count} 件")
    print(f"  - 所要時間: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分)")
    print(f"  - インデックス保存先: {INDEX_FILE}")
    print(f"  - プリセット保存先: {PRESET_DIR}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Batch Physical Sound Internet Crawler & Analyzer")
    parser.add_argument("--lang", type=str, default="ALL", choices=["ALL", "JP", "KR", "AF"], help="Filter language")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to process (0 = all)")
    parser.add_argument("--delay", type=float, default=0.6, help="Delay between web requests in seconds")
    parser.add_argument("--api-key", type=str, default="", help="Freesound API token (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing presets")
    args = parser.parse_args()

    run_batch_crawler(
        lang=args.lang,
        limit=args.limit,
        delay=args.delay,
        api_key=args.api_key,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
