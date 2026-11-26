"""Write a synthetic Brandwatch export and its media to data/raw/ so the pipeline runs before real data arrives.

Every account, post and number is invented. Media files are pre-placed in data/raw/media/,
so the export box never downloads the fake URLs. Videos need macOS `say` (voice Damayanti) and ffmpeg.

Run: uv run python scripts/make_dummy_data.py
"""

import random
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MEDIA = RAW / "media"
START = datetime(2025, 2, 17, 8, 0, tzinfo=timezone.utc)

# Header of the real Brandwatch export, in order. Columns the dummy does not fill stay empty.
BRANDWATCH_COLUMNS = [
    "Query Id", "Query Name", "Date", "Title", "Url", "Domain", "Sentiment", "Page Type", "Language", "Country Code",
    "Continent Code", "Continent", "Country", "City Code", "Account Type", "Added", "Assignment", "Author",
    "Category Details", "Checked", "City", "Display URLs", "Entity Info", "Expanded URLs", "Facebook Author ID",
    "Facebook Comments", "Facebook Likes", "Facebook Role", "Facebook Shares", "Facebook Subtype", "Full Name",
    "Full Text", "Gender", "Impressions", "Instagram Comments", "Instagram Followers", "Instagram Following",
    "Instagram Interactions Count", "Instagram Posts", "Interest", "Last Assignment Date", "Latitude",
    "Location Name", "Longitude", "Media Filter", "Media URLs", "Mentioned Authors", "Original Url", "Priority",
    "Professions", "Resource Id", "Short URLs", "Starred", "Station Name", "Viewership", "Status", "Subtype",
    "Thread Author", "Thread Created Date", "Thread Entry Type", "Thread Id", "Thread URL", "Total Monthly Visitors",
    "X Author ID", "X Channel Role", "X Followers", "X Following", "X Replies", "X Reply to", "X Repost of",
    "X Reposts", "X Likes", "X Posts", "X Verified", "Updated", "Reach (new)", "Potential Audience", "Social Shares",
    "Publication Name", "Licenses", "Redacted", "Redacted Fields", "Redaction Reason", "Asset Content Id",
    "Asset Thumb Id", "Author Verified Type", "Avatar", "Batch Id", "Blog Name", "Broadcast Media Url",
    "Is Syndicated", "Air Type", "Broadcast Type", "Media Type", "Ad Value", "Circulation", "Region", "Region Code",
    "Daily Visitors", "Engagement Type", "Hashtags", "Item Review", "Kicker", "Linkedin Comments",
    "Linkedin Engagement", "Linkedin Impressions", "Linkedin Likes", "Linkedin Shares", "Linkedin Sponsored",
    "Linkedin Video Views", "Parent Post Id", "Parent Blog Name", "Pub Type", "Publisher Sub Type", "Rating",
    "Reddit Author Karma", "Subreddit", "Root Post Id", "Root Blog Name", "Subscriptions", "Sub Title",
    "React Score Overall", "React Score Emotionality", "React Score Harmful", "Engagement Score", "Publication Id",
    "Page Type Name", "Content Source", "Content Source Name", "Likes", "Comments", "Shares", "Views", "Reactions",
    "Custom", "Ad Currency Code", "Bluesky Author Id", "Bluesky Followers", "Bluesky Following", "Bluesky Likes",
    "Bluesky Posts", "Bluesky Quotes", "Bluesky Replies", "Bluesky Reposts", "Copyright", "Factiva Attribute Code",
    "Has Full Text", "Image Info", "Impact", "Instagram Likes", "Mention Id", "Podcast Audience Estimate",
    "Podcast Duration Ms", "Raw Metadata", "Reportable", "Resource Views", "Threads Likes", "Threads Quotes",
    "Threads Replies", "Threads Reposts", "Threads Shares", "Threads Views", "Tiktok Comments",
    "Tiktok Connected Account", "Tiktok Likes", "Tiktok Reach", "Tiktok Shares", "Tiktok Views",
    "Unified Source Name", "Weblog Title", "Youtube Comments", "Youtube Duration Milliseconds", "Youtube Favourites",
    "Youtube Likes", "Youtube Subscriber Count", "Youtube Video Count", "Emotion",
]

ACCOUNTS = [
    # handle, kind, followers, following, verified
    ("kabarnusantara", "media", 850_000, 300, True),
    ("beritaterkini_id", "media", 420_000, 150, True),
    ("grok", "grok", 2_000_000, 10, True),
    *[(f"suaraRakyat{i:02d}", "buzzer", 120 + 15 * i, 4_900, False) for i in range(1, 7)],
    *[(f"warga_{name}", "organic", 300 + 97 * i, 280 + 31 * i, False) for i, name in enumerate(
        ["andi", "sari", "budi", "dewi", "rizky", "putri", "agus", "nina", "fajar", "lina", "yoga", "maya"]
    )],
]
AUTHOR_ID = {handle: str(1_000_000 + i) for i, (handle, *_rest) in enumerate(ACCOUNTS)}

HOAX_TEXTS = [
    "Video tentara menembaki mahasiswa saat demo #IndonesiaGelap di Jakarta, sebarkan sebelum dihapus!",
    "Presiden kabur ke luar negeri setelah demo #IndonesiaGelap, istana kosong sejak kemarin",
    "Pemerintah resmi hapus beasiswa KIP Kuliah mulai bulan depan #IndonesiaGelap",
    "Foto Monas dibakar massa aksi #IndonesiaGelap tadi malam",
    "Rekaman suara menteri bilang rakyat miskin tidak usah kuliah #IndonesiaGelap",
]
ORGANIC_TEXTS = [
    "Aksi #IndonesiaGelap hari ini berjalan damai, mahasiswa menyampaikan tuntutan soal pendidikan",
    "Kenapa anggaran pendidikan dipotong? Kami butuh penjelasan #IndonesiaGelap",
    "Ikut aksi #IndonesiaGelap di Bandung, ramai tapi tertib",
    "Tuntutan #IndonesiaGelap: evaluasi efisiensi anggaran dan transparansi kebijakan",
    "Semoga suara mahasiswa didengar pemerintah #IndonesiaGelap",
    "Liputan langsung aksi #IndonesiaGelap dari depan gedung DPR",
]
BUZZER_TEXT = "#IndonesiaGelap itu provokasi asing, jangan mau diadu domba! Dukung pemerintah! cek saweria.co/dukungnkri"
REPLIES = [
    "Ini hoaks, sudah dibantah polisi",
    "Sumbernya dari mana? Kok tidak ada di berita resmi",
    "Kejadiannya kemarin sore, saya lihat sendiri",
    "Jangan asal sebar, cek dulu faktanya",
]
GROK_REPLIES = [
    "Gambar ini kemungkinan besar dibuat dengan AI: ada distorsi pada tangan dan teks.",
    "Tidak ada indikasi bahwa media ini hasil rekayasa AI; ini rekaman asli dari lokasi.",
]


def write_image(path: Path, caption: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (640, 480), tuple(int(c) for c in rng.integers(30, 200, 3)))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    for line_number, start in enumerate(range(0, len(caption), 34)):
        draw.text((24, 160 + 40 * line_number), caption[start : start + 34], fill="white", font=font)
    image.save(path, "JPEG", quality=90)


def write_video(path: Path, speech: str, seed: int) -> None:
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as work:
        frames_path = Path(work) / "frames.mp4"
        audio_path = Path(work) / "speech.aiff"
        writer = cv2.VideoWriter(str(frames_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
        rng = np.random.default_rng(seed)
        for frame_number in range(40):
            frame = np.full((360, 640, 3), rng.integers(20, 120, 3), dtype=np.uint8)
            cv2.putText(frame, "INDONESIA GELAP", (90, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            cv2.putText(frame, f"{frame_number:02d}", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            writer.write(frame)
        writer.release()
        subprocess.run(["say", "-v", "Damayanti", "-o", str(audio_path), speech], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(frames_path), "-i", str(audio_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
            check=True,
        )


def main() -> None:
    random.seed(42)
    MEDIA.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    account = {handle: (kind, followers, following, verified) for handle, kind, followers, following, verified in ACCOUNTS}
    by_kind = {kind: [h for h, (k, *_r) in account.items() if k == kind] for kind in ("media", "buzzer", "organic")}
    rows = []
    created_at = {}
    next_id = iter(range(1_900_000_000_000_000_000, 2_000_000_000_000_000_000))

    def add(handle, text, created, engagement=None, parent=None, media_type=None, likes=0, reposts=0, replies=0):
        post_id = str(next(next_id))
        kind, followers, following, verified = account[handle]
        url = f"https://x.com/{handle}/status/{post_id}"
        media_url = ""
        if media_type == "photo":
            media_url = f"https://pbs.twimg.com/media/{post_id}.jpg"
            write_image(MEDIA / f"{post_id}.jpg", text, int(post_id) % 10_000)
        if media_type == "video":
            media_url = f"https://video.twimg.com/ext_tw_video/{post_id}.mp4"
            write_video(MEDIA / f"{post_id}.mp4", text.replace("#", ""), int(post_id) % 10_000)
        parent_url = parent["Url"] if parent is not None else ""
        rows.append({
            "Query Id": 2003456789,
            "Query Name": "IndonesiaGelap X",
            "Date": created.strftime("%Y-%m-%d %H:%M:%S.0"),
            "Title": text[:80],
            "Url": url,
            "Domain": "x.com",
            "Sentiment": random.choice(["negative", "neutral", "positive"]),
            "Language": "id",
            "Country Code": "id",
            "Author": handle,
            "Full Name": handle.replace("_", " ").title(),
            "Full Text": text,
            "Thread Entry Type": {None: "post", "REPLY": "reply", "RETWEET": "share", "QUOTE": "post"}[engagement],
            "Thread Id": (parent["Thread Id"] if parent is not None else post_id),
            "Engagement Type": engagement or "",
            "X Author ID": AUTHOR_ID[handle],
            "X Followers": followers + random.randint(0, 40) * (len(rows) // 20),
            "X Following": following,
            "X Posts": random.randint(500, 90_000),
            "X Verified": verified,
            "X Replies": replies,
            "X Reposts": reposts,
            "X Likes": likes,
            "X Reply to": parent_url if engagement == "REPLY" else "",
            "X Repost of": parent_url if engagement == "RETWEET" else "",
            "Original Url": url,  # the real export puts the post's own URL here
            "Subtype": media_type or "status",
            "Media URLs": media_url,
            "Mentioned Authors": f"@{parent['Author']}" if parent is not None else "",
            "Hashtags": "#IndonesiaGelap",
            "Expanded URLs": parent_url if engagement == "QUOTE" else (
                "https://saweria.co/dukungnkri" if "saweria.co" in text else ""
            ),
            "Impressions": followers * random.randint(1, 3),
            "Reach (new)": followers * random.randint(1, 2),
            "Account Type": "organisational" if kind == "media" else "individual",
            "Resource Id": f"bw-{post_id}",
            "Content Source Name": "X",
        })
        created_at[url] = created
        return rows[-1]

    originals = []
    for i, text in enumerate(HOAX_TEXTS):
        media_type = ["video", "status", "status", "photo", "video"][i]
        originals.append(add(random.choice(by_kind["organic"]), text, START + timedelta(hours=3 * i),
                             media_type=media_type, likes=random.randint(200, 3000), reposts=random.randint(100, 900),
                             replies=random.randint(10, 200)))
    for i, text in enumerate(ORGANIC_TEXTS):
        media_type = "photo" if i % 3 == 0 else "status"
        handle = by_kind["media"][i % 2] if i == 5 else random.choice(by_kind["organic"])
        originals.append(add(handle, text, START + timedelta(hours=2 * i + 1), media_type=media_type,
                             likes=random.randint(20, 900), reposts=random.randint(5, 300), replies=random.randint(1, 80)))
    # Buzzers post the same text in a burst, a few minutes apart.
    burst_start = START + timedelta(hours=10)
    for i, handle in enumerate(by_kind["buzzer"] * 2):
        originals.append(add(handle, BUZZER_TEXT, burst_start + timedelta(minutes=3 * i),
                             likes=random.randint(0, 15), reposts=random.randint(0, 5)))
    for parent in originals[:11]:
        for j in range(2):
            add(random.choice(by_kind["organic"]), random.choice(REPLIES),
                created_at[parent["Url"]] + timedelta(minutes=20 + j),
                engagement="REPLY", parent=parent)
        if parent["Subtype"] != "status":
            add("grok", random.choice(GROK_REPLIES),
                created_at[parent["Url"]] + timedelta(minutes=45),
                engagement="REPLY", parent=parent)
    for parent in originals[:5]:
        for handle in random.sample(by_kind["buzzer"], 3):
            add(handle, f"RT @{parent['Author']}: {parent['Full Text']}",
                created_at[parent["Url"]] + timedelta(minutes=5),
                engagement="RETWEET", parent=parent)
        add(random.choice(by_kind["organic"]), "Ini perlu dicek kebenarannya #IndonesiaGelap",
            created_at[parent["Url"]] + timedelta(hours=1),
            engagement="QUOTE", parent=parent)

    # Non-X sources the Brandwatch query also returns; x_only must drop them.
    for source, domain, url, text in [
        ("News", "detik.com", "https://news.detik.com/berita/d-7790001/aksi-indonesia-gelap-di-jakarta",
         "Aksi Indonesia Gelap digelar mahasiswa di Jakarta"),
        ("Facebook", "facebook.com", "https://www.facebook.com/groups/123/posts/456",
         "Ayo ikut aksi #IndonesiaGelap besok"),
        ("TikTok", "tiktok.com", "https://www.tiktok.com/@warga/video/7470000000000000000",
         "Suasana aksi #IndonesiaGelap #fyp"),
    ]:
        rows.append({
            "Query Id": 2003456789, "Query Name": "IndonesiaGelap X", "Date": START.strftime("%Y-%m-%d %H:%M:%S.0"),
            "Title": text, "Url": url, "Domain": domain, "Language": "id", "Author": "nonx_author",
            "Full Text": text, "Content Source Name": source, "Subtype": "status",
        })

    assert set().union(*rows) <= set(BRANDWATCH_COLUMNS), set().union(*rows) - set(BRANDWATCH_COLUMNS)
    # The real "Bulk Mentions Download" starts with 5 report rows and a blank line above the column header.
    with (RAW / "brandwatch_export.csv").open("w", encoding="utf-8") as csv_file:
        csv_file.write('"Report:","Bulk Mentions Download"\n"Brand:","Dummy"\n"From:",""\n"To:",""\n"Label:",""\n\n')
        pd.DataFrame(rows).reindex(columns=BRANDWATCH_COLUMNS).to_csv(csv_file, index=False)
    print(f"Wrote {len(rows)} posts to {RAW / 'brandwatch_export.csv'}")


if __name__ == "__main__":
    main()
