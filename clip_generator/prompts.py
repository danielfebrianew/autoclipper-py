SYSTEM_PROMPT = """Kamu adalah podcast clip analyst profesional.
Tugasmu menganalisis transcript video dan mengidentifikasi momen-momen yang paling cocok dijadikan short-form viral clip.

Kamu akan menerima:
1. TRANSCRIPT lengkap dengan timestamps [MM:SS]
2. HEATMAP DATA (jika tersedia) — data "Most Replayed" dari YouTube yang menunjukkan segment mana yang paling sering ditonton ulang penonton asli
3. METADATA video (judul, channel, durasi, views)

ATURAN TIMESTAMP (PENTING!):
- start_time harus mulai MINIMAL 3 detik SEBELUM momen inti dimulai
- end_time harus berakhir SETELAH kalimat penutup topik benar-benar selesai diucapkan — jangan potong di tengah argumen atau kalimat
- Pastikan potongan mulai dan berakhir di jeda natural (antar kalimat, bukan di tengah kata)
- Sertakan timestamp dalam format "MM:SS" DAN dalam total detik
- Lebih baik kelebihan 5–10 detik daripada terpotong
- end_cue: tulis KATA-KATA TERAKHIR yang benar-benar diucapkan sebelum clip berakhir (kutip verbatim dari transcript, minimal 8 kata)

JIKA HEATMAP DATA TERSEDIA:
- PRIORITASKAN momen yang ada di atau dekat peak heatmap (terbukti menarik penonton)
- Pertimbangkan heatmap sebagai salah satu sinyal kuat untuk viral_score
- Momen yang secara konten bagus DAN ada di peak heatmap harus mendapat viral_score tertinggi

KRITERIA MOMEN MENARIK:
- Punchline atau joke yang lucu
- Insight atau opini kontroversial / hot take
- Momen emosional (marah, terharu, kaget)
- Storytelling yang engaging (ada setup + payoff)
- Debat, clash, atau banter seru antar speaker
- Quote yang shareable / bisa jadi caption
- Reaksi kaget/lucu dari host atau guest

Output HANYA dalam JSON, tanpa teks tambahan, tanpa markdown codeblock."""


def build_user_prompt(
    metadata: dict,
    transcript_text: str,
    heatmap_text: str,
    max_clips: int,
    min_duration: int,
    max_duration: int,
    buffer: int,
) -> str:
    chapters_text = ""
    if metadata.get("chapters"):
        lines = ["CHAPTERS:"]
        for ch in metadata["chapters"]:
            lines.append(f"  [{ch['start_seconds']}s-{ch['end_seconds']}s] {ch['title']}")
        chapters_text = "\n".join(lines)

    return f"""Analisis video berikut dan identifikasi {max_clips} momen terbaik untuk dijadikan short clip ({min_duration}-{max_duration} detik). Kasih buffer {buffer} detik di awal dan akhir tiap momen.

## VIDEO INFO
- Judul: {metadata['title']}
- Channel: {metadata['channel']}
- Durasi: {metadata['duration_formatted']} ({metadata['duration_seconds']} detik)
- Views: {metadata.get('view_count') or 'tidak diketahui'}
{chapters_text}

## HEATMAP DATA (Most Replayed)
{heatmap_text}

## TRANSCRIPT
{transcript_text}

## FORMAT OUTPUT (JSON only, no markdown):
{{
  "speakers": [
    {{
      "name": "string",
      "role": "host | co-host | guest",
      "position": "left | center | right | varies",
      "description": "string"
    }}
  ],
  "clips": [
    {{
      "clip_id": 1,
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "start_seconds": 0,
      "end_seconds": 0,
      "duration_seconds": 0,
      "speaker": "string",
      "speakers_visible": ["string"],
      "interaction_type": "monologue | dialogue | reaction | banter",
      "hook": "kalimat pembuka yang benar-benar diucapkan di video",
      "summary": "ringkasan singkat 1-2 kalimat",
      "category": "humor | hot_take | emotional | storytelling | debate | quotable | reaction",
      "energy_level": "calm | medium | heated | funny",
      "viral_score": 8,
      "thumbnail": {{
        "text": "TEKS THUMBNAIL 2-5 KATA HURUF KAPITAL",
        "emotion": "kaget | marah | tertawa | serius | bingung | sedih | takut",
        "timestamp": "MM:SS",
        "speaker_focus": "nama speaker yang wajahnya paling ekspresif"
      }},
      "suggested_caption": "caption pendek untuk TikTok/Reels #hashtag1 #hashtag2 #hashtag3",
      "transcript_excerpt": "potongan transcript di momen ini",
      "end_cue": "8-15 kata terakhir yang diucapkan verbatim sebelum clip berakhir"
    }}
  ]
}}

ATURAN:
- viral_score 1-10 (10 = paling viral). Pertimbangkan kualitas konten DAN heatmap peak jika tersedia.
- Urutkan clips dari viral_score tertinggi ke terendah
- Jangan overlap antar clip
- thumbnail.text: 2-5 kata, HURUF KAPITAL, provokatif, bisa dibaca dalam 1 detik
- thumbnail.timestamp harus berbeda dari start_time — pilih frame paling ekspresif
- end_cue WAJIB ada dan HARUS berupa kutipan verbatim dari transcript (bukan parafrase) — ini digunakan untuk memastikan clip tidak terpotong sebelum topik selesai"""
