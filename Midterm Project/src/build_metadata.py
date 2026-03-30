from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata
import pandas as pd
import re


def find_excel_files(dataset_root: Path) -> list[Path]:
    return sorted(list(dataset_root.rglob("*.xlsx")) + list(dataset_root.rglob("*.xls")))


def parse_gender_from_code(code: str) -> str:
    c = code.upper().strip()
    # Typical codes used in this dataset:
    # - Turkish: E/K/C
    # - English letters: M/F/C
    if c in {"E", "M"}:
        return "Male"
    if c in {"K", "F"}:
        return "Female"
    if c == "C":
        return "Child"
    return "Unknown"


def build_metadata_from_wav_names(dataset_root: Path) -> pd.DataFrame:
    wav_files = sorted(dataset_root.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {dataset_root}")

    rows: list[dict[str, object]] = []
    for wav in wav_files:
        stem_parts = wav.stem.split("_")
        gender_code = stem_parts[2] if len(stem_parts) > 2 else ""
        age_raw = stem_parts[3] if len(stem_parts) > 3 else ""
        age = int(age_raw) if age_raw.isdigit() else None

        rows.append(
            {
                "File_Name": wav.name,
                "Gender_Code": gender_code,
                "Gender": parse_gender_from_code(gender_code),
                "Age": age,
                "Audio_Path": str(wav),
                "Audio_Exists": True,
                "Source_Excel": "",
            }
        )

    return pd.DataFrame(rows)


def find_audio_by_name(dataset_root: Path, file_name: str) -> Path | None:
    matches = list(dataset_root.rglob(file_name))
    return matches[0] if matches else None


def normalize_for_match(s: str) -> str:
    """
    Filename matching is sensitive to unicode normalization (e.g., Turkish characters).
    This function normalizes unicode, strips diacritics, and makes comparison case-insensitive.
    """
    s = str(s).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii", errors="ignore")
    s = s.replace(" ", "")
    s = s.upper()
    s = s.rstrip("._")

    # Canonicalize numeric tokens to avoid mismatches like "09" vs "9".
    parts = s.split("_")
    new_parts: list[str] = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            new_parts.append(str(int(p)))
            continue
        # e.g. C01 -> C1
        if len(p) >= 2 and p[0].isalpha() and p[1:].isdigit():
            new_parts.append(p[0] + str(int(p[1:])))
            continue
        # e.g. D04 -> D4 (also covers E03/K02 etc.)
        # token like <letters><leadingzeros><digits>
        import re

        m = re.fullmatch(r"([A-Z]+)0+(\d+)", p)
        if m:
            new_parts.append(m.group(1) + str(int(m.group(2))))
            continue
        new_parts.append(p)

    return "_".join(new_parts)


def map_feeling_token(feeling_token: str) -> list[str]:
    """
    Return alternative feeling tokens to increase filename matching robustness.
    Both English and Turkish dataset spellings appear across groups.
    """
    ft = normalize_for_match(feeling_token)
    mapping: dict[str, list[str]] = {
        "NEUTRAL": ["NOTR", "NATR", "NEUTRAL"],
        "FURIOUS": ["OFKELI", "OFKELİ", "OFKELİ", "FURIOUS"],
        "HAPPY": ["MUTLU", "HAPPY"],
        "SAD": ["UZGUN", "MUTSUZ", "SAD"],
        "SHOCKED": ["SASKIN", "SASIRMA", "SHOCKED"],
        "SURPRISED": ["SASKIN", "SHOCKED", "SURPRISED"],
        # In some files emotion might be "ANGRY"
        "ANGRY": ["FURIOUS", "ANGRY"],
    }
    alts = mapping.get(ft, [])
    # Always keep original token as last resort.
    return [feeling_token] + alts


def map_gender_token_for_filename(gender_token: str) -> list[str]:
    """
    Dataset uses multiple gender coding conventions across groups.
    Examples observed in WAV filenames:
    - E/K/C and M/F/C (both encode Male/Female/Child)
    """
    gt = normalize_for_match(gender_token)
    mapping: dict[str, list[str]] = {
        "F": ["F", "K"],
        "K": ["F", "K"],
        "M": ["M", "E"],
        "E": ["M", "E"],
        "C": ["C"],
    }
    alts = mapping.get(gt, [gender_token])
    # Preserve original token at least once
    if gender_token not in alts:
        alts = [gender_token] + alts
    return alts


def build_wav_indexes(dataset_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    wav_files = list(dataset_root.rglob("*.wav"))
    wav_by_name: dict[str, Path] = {}
    wav_by_stem: dict[str, Path] = {}
    # Relaxed indices for fallback matching.
    # Entries are stored as dicts so we can score candidates by similarity.
    by_gd_age_sent: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
    by_gd_age: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    by_gd_sent: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    by_gd: dict[tuple[int, int], list[dict[str, object]]] = {}
    all_entries: list[dict[str, object]] = []

    for wav in wav_files:
        n = normalize_for_match(wav.name)
        st = normalize_for_match(wav.stem)
        if n and n not in wav_by_name:
            wav_by_name[n] = wav
        if st and st not in wav_by_stem:
            wav_by_stem[st] = wav

        parsed = parse_filename_parts(normalize_for_match(wav.stem))
        if parsed is None:
            continue

        entry = {
            "path": wav,
            **parsed,
        }
        all_entries.append(entry)

        key_exact = (parsed["G"], parsed["D"], parsed["Age"], parsed["Sentence"])
        by_gd_age_sent.setdefault(key_exact, []).append(entry)

        key_age = (parsed["G"], parsed["D"], parsed["Age"])
        by_gd_age.setdefault(key_age, []).append(entry)

        key_sent = (parsed["G"], parsed["D"], parsed["Sentence"])
        by_gd_sent.setdefault(key_sent, []).append(entry)

        key_gd = (parsed["G"], parsed["D"])
        by_gd.setdefault(key_gd, []).append(entry)

    # Return type updated: caller expects at least wav_by_name/wav_by_stem.
    # Extra indices are included as additional return values.
    return wav_by_name, wav_by_stem, by_gd_age_sent, by_gd_age, by_gd_sent, by_gd, all_entries


def parse_filename_parts(normalized_stem: str) -> dict[str, object] | None:
    """
    Parse normalized stem into (G, D, GenderCode, Age, Feeling, Sentence).
    normalized_stem must already be produced by `normalize_for_match`.
    """
    if not normalized_stem:
        return None
    tokens = [t for t in normalized_stem.split("_") if t]
    if len(tokens) < 4:
        return None

    g = None
    for t in tokens:
        m = re.fullmatch(r"G(\d+)", t)
        if m:
            g = int(m.group(1))
            break
    if g is None:
        return None

    # Sentence is last token like C1 / C4
    sentence = None
    for t in reversed(tokens):
        m = re.fullmatch(r"C(\d+)", t)
        if m:
            sentence = int(m.group(1))
            break
    if sentence is None:
        return None

    # Find first token starting with D (could be D04 or D03K)
    d_idx = None
    d_token = None
    for i, t in enumerate(tokens):
        if t.startswith("D"):
            d_idx = i
            d_token = t
            break
    if d_idx is None or d_token is None:
        return None

    m = re.fullmatch(r"D(\d+)([A-Z]?)", d_token)
    if not m:
        m2 = re.match(r"D(\d+)", d_token)
        if not m2:
            return None
        d = int(m2.group(1))
        d_gender_embedded = ""
    else:
        d = int(m.group(1))
        d_gender_embedded = m.group(2) or ""

    # Tokens between D-token and Sentence token
    sent_idx = max(i for i, t in enumerate(tokens) if re.fullmatch(r"C\d+", t))
    middle = tokens[d_idx + 1 : sent_idx]
    if not middle:
        return None

    # Infer gender/age/feeling using common filename patterns.
    gender = None
    age = None
    feeling = None

    def is_gender_code(t: str) -> bool:
        return t in {"E", "K", "M", "F", "C"}

    if d_gender_embedded and len(middle) >= 2:
        # Pattern: G##_D##<gender>_<age>_<feeling>_C#
        gender = d_gender_embedded
        # age is first numeric token
        for t in middle:
            if t.isdigit():
                age = int(t)
                break
        if age is None:
            return None
        # feeling is token just after the age in middle, otherwise first non-numeric token
        idx_age = next((i for i, t in enumerate(middle) if t.isdigit()), None)
        if idx_age is not None and idx_age + 1 < len(middle):
            feeling = middle[idx_age + 1]
        else:
            feeling = next((t for t in middle if not t.isdigit()), None)
    else:
        # Pattern: G##_D##_<gender>_<age>_<feeling>_C#
        if len(middle) >= 3 and is_gender_code(middle[0]):
            gender = middle[0]
            # age: first numeric after gender
            for t in middle[1:]:
                if t.isdigit():
                    age = int(t)
                    break
            if age is None:
                return None
            # feeling token: first non-numeric after age
            # find first numeric position
            idx_age = next((i for i, t in enumerate(middle) if t.isdigit()), None)
            if idx_age is not None and idx_age + 1 < len(middle):
                feeling = middle[idx_age + 1]
            else:
                feeling = next((t for t in middle if not t.isdigit()), None)
        else:
            # Fallback: infer age from first numeric in middle; gender from previous token; feeling from next
            idx_age = next((i for i, t in enumerate(middle) if t.isdigit()), None)
            if idx_age is None:
                return None
            age = int(middle[idx_age])
            if idx_age - 1 >= 0 and is_gender_code(middle[idx_age - 1]):
                gender = middle[idx_age - 1]
            feeling = middle[idx_age + 1] if idx_age + 1 < len(middle) else None

    if gender is None or age is None or feeling is None:
        return None

    return {
        "G": g,
        "D": d,
        "GenderCode": gender,
        "Age": age,
        "Feeling": feeling,
        "Sentence": sentence,
    }


def resolve_audio_path(
    file_name: str,
    wav_by_name: dict[str, Path],
    wav_by_stem: dict[str, Path],
    by_gd_age_sent: dict[tuple[int, int, int, int], list[dict[str, object]]],
    by_gd_age: dict[tuple[int, int, int], list[dict[str, object]]],
    by_gd_sent: dict[tuple[int, int, int], list[dict[str, object]]],
    by_gd: dict[tuple[int, int], list[dict[str, object]]],
    all_entries: list[dict[str, object]],
) -> str:
    if not file_name:
        return ""
    raw = str(file_name).strip()
    candidates = [raw]
    if not raw.lower().endswith(".wav"):
        candidates.append(raw + ".wav")

    for cand in candidates:
        nn = normalize_for_match(cand)
        if nn and nn in wav_by_name:
            return str(wav_by_name[nn])

        # Also try matching by stem (without extension)
        if cand.lower().endswith(".wav"):
            stem = cand[:-4]
        else:
            stem = cand
        ns = normalize_for_match(stem)
        if ns and ns in wav_by_stem:
            return str(wav_by_stem[ns])

    # If direct resolution fails, try feeling-token substitutions.
    # Typical pattern: Gxx_Dyy_<Gender>_<Age>_<Feeling>_C<Sentence>.wav
    stem = raw[:-4] if raw.lower().endswith(".wav") else raw
    norm_stem = normalize_for_match(stem)
    tokens = stem.split("_")
    if len(tokens) >= 5:
        # Common format has:
        # [Gxx, Dyy, Gender, Age, Feeling, C#]
        # but some files may have variable token lengths; we only apply
        # gender/feeling substitutions when we can clearly identify positions.
        feeling_token = tokens[-2]
        alt_feelings = map_feeling_token(feeling_token)
        alt_genders = [tokens[2]] if len(tokens) < 4 else map_gender_token_for_filename(tokens[2])

        for alt_gender in alt_genders:
            for alt_feeling in alt_feelings:
                alt_tokens = tokens[:]
                if len(tokens) >= 6:
                    # Replace gender token at index 2
                    alt_tokens[2] = alt_gender
                # Replace feeling token at index -2
                alt_tokens[-2] = alt_feeling

                alt_stem = "_".join(alt_tokens)
                alt_name = alt_stem + (".wav" if not raw.lower().endswith(".wav") else "")

                for alt in [alt_name, alt_stem]:
                    nn = normalize_for_match(alt)
                    if nn and nn in wav_by_name:
                        return str(wav_by_name[nn])
                    ns = (
                        normalize_for_match(alt[:-4])
                        if alt.lower().endswith(".wav")
                        else normalize_for_match(alt)
                    )
                    if ns and ns in wav_by_stem:
                        return str(wav_by_stem[ns])

    # Last resort: relaxed structured matching.
    # Even if exact filename tokens differ, we try to pick the closest wav candidate
    # based on (G, D, Age, Sentence) and (Gender, Feeling) similarity.
    stem = raw[:-4] if raw.lower().endswith(".wav") else raw
    norm_stem = normalize_for_match(stem)
    parsed_excel = parse_filename_parts(norm_stem)
    if parsed_excel is not None:
        G = int(parsed_excel["G"])
        D = int(parsed_excel["D"])
        age = int(parsed_excel["Age"])
        sent = int(parsed_excel["Sentence"])
        excel_gender = str(parsed_excel["GenderCode"])
        excel_feeling = str(parsed_excel["Feeling"])

        # Build allowed sets for scoring
        excel_gender_alts = set(map(normalize_for_match, map_gender_token_for_filename(excel_gender)))
        excel_feeling_alts = set(map(normalize_for_match, map_feeling_token(excel_feeling)))

        # Candidate pools by relaxing constraints from strictest to loosest.
        candidate_pools: list[list[dict[str, object]]] = []
        key_exact = (G, D, age, sent)
        key_age = (G, D, age)
        key_sent = (G, D, sent)
        key_gd = (G, D)

        candidate_pools.append(by_gd_age_sent.get(key_exact, []))
        candidate_pools.append(by_gd_age.get(key_age, []))
        candidate_pools.append(by_gd_sent.get(key_sent, []))
        candidate_pools.append(by_gd.get(key_gd, []))
        candidate_pools.append(all_entries)

        best: tuple[float, Path] | None = None

        for pool in candidate_pools:
            if not pool:
                continue

            for cand in pool:
                cand_gender = str(cand["GenderCode"])
                cand_feeling = str(cand["Feeling"])
                cand_age = int(cand["Age"])
                cand_sent = int(cand["Sentence"])

                score = 0.0
                if cand_age == age:
                    score += 4.0
                if cand_sent == sent:
                    score += 3.0
                if normalize_for_match(cand_gender) in excel_gender_alts:
                    score += 2.0
                if normalize_for_match(cand_feeling) in excel_feeling_alts:
                    score += 1.0

                path = cand["path"]  # type: ignore[assignment]
                if not isinstance(path, Path):
                    continue

                if best is None or score > best[0]:
                    best = (score, path)

            # If we found at least one candidate in this pool, stop relaxing further.
            if best is not None:
                break

        if best is not None:
            return str(best[1])

    return ""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Common column name normalization for mixed group templates.
    mapping = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in {"gender", "cinsiyet"}:
            mapping[col] = "Gender"
        elif c in {"age", "yas"}:
            mapping[col] = "Age"
        elif c in {"subject_id", "subject id", "subjectid", "subject"}:
            mapping[col] = "Subject_ID"
        elif c in {"feeling", "emotion", "duygu"}:
            mapping[col] = "Feeling"
        elif c in {"sentence_no", "sentence no", "sentenceno", "sentence"}:
            mapping[col] = "Sentence_No"
        elif c in {"file", "filename", "file_name", "wav", "audio"} or c.replace(" ", "") in {
            "filename",
            "fileName",
        } or c in {"file name", "dosya adı", "dosya adi"}:
            mapping[col] = "File_Name"
    return df.rename(columns=mapping)


def infer_columns_by_content(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    For XLSX files where headers are missing/broken (all columns become 'Unnamed'),
    infer which column corresponds to File_Name/Gender/Age/Feeling/Subject_ID/Sentence_No
    by looking at cell contents.
    """
    if df_raw is None or df_raw.empty:
        return None

    ncols = df_raw.shape[1]
    wav_cols: list[int] = []
    gender_cols: list[int] = []
    subject_cols: list[int] = []
    feeling_cols: list[int] = []
    numeric_cols: list[int] = []

    feeling_keywords = {
        "ANGRY",
        "NEUTRAL",
        "FURIOUS",
        "HAPPY",
        "SAD",
        "SHOCKED",
        "MUTLU",
        "UZGUN",
        "OFKELI",
        "SASKIN",
        "SASIRMA",
        "MUTSUZ",
        "ŞAŞKıN",
        "ŞAŞKIN",
    }

    for i in range(ncols):
        s = df_raw.iloc[:, i].dropna()
        if s.empty:
            continue

        as_str = s.astype(str).head(200)
        up = as_str.str.upper()

        # Contains .wav
        if up.str.contains(r"\.WAV$", regex=True, na=False).any():
            wav_cols.append(i)

        # Gender codes
        gender_candidates = set(
            x.strip().upper() for x in up.tolist() if x.strip().upper() in {"M", "F", "C", "E", "K"}
        )
        if gender_candidates:
            gender_cols.append(i)

        # Subject_ID looks like D01, D1, etc.
        if up.str.match(r"^D\d+$", na=False).any():
            subject_cols.append(i)

        # Feeling keywords
        if any(tok in feeling_keywords for tok in set(norm for norm in [x.strip().upper() for x in up.tolist()])):
            # fallback: do a simpler contains check
            pass
        if up.str.contains("|".join(list(feeling_keywords)), regex=True, na=False).any():
            feeling_cols.append(i)

        # Numeric columns
        # Age/sentence are stored as numbers; detect columns that have many ints.
        numeric_vals: list[int] = []
        for v in as_str.tolist():
            vv = str(v).strip()
            try:
                # float->int round is OK for dataset metadata
                numeric_vals.append(int(float(vv)))
            except Exception:
                continue
        if len(numeric_vals) >= 5:
            numeric_cols.append(i)

    # Basic picks
    file_col = wav_cols[0] if wav_cols else None
    gender_col = gender_cols[0] if gender_cols else None
    subject_col = subject_cols[0] if subject_cols else None
    feeling_col = feeling_cols[0] if feeling_cols else None

    # Decide Age vs Sentence from numeric columns:
    # Sentence_No usually small (<=10), Age larger (>=10 typically 11..50)
    age_col = None
    sent_col = None
    for i in numeric_cols:
        s = df_raw.iloc[:, i].dropna().astype(str)
        vals: list[int] = []
        for v in s.tolist()[:200]:
            try:
                vals.append(int(float(v.strip())))
            except Exception:
                continue
        if not vals:
            continue
        vmax = max(vals)
        vmed = sorted(vals)[len(vals) // 2]
        # Sentence: median <= 10 and max <= 10..12
        if vmed <= 10 and vmax <= 12:
            sent_col = i if sent_col is None else sent_col
        # Age: median > 10 and max >= 15
        if vmed > 10 and vmax >= 15:
            age_col = i if age_col is None else age_col

    if file_col is None or gender_col is None:
        return None

    # Create mapped df with standard column names.
    out = pd.DataFrame()
    out["File_Name"] = df_raw.iloc[:, file_col]
    out["Gender"] = df_raw.iloc[:, gender_col]
    if age_col is not None:
        out["Age"] = df_raw.iloc[:, age_col]
    if feeling_col is not None:
        out["Feeling"] = df_raw.iloc[:, feeling_col]
    if subject_col is not None:
        out["Subject_ID"] = df_raw.iloc[:, subject_col]
    if sent_col is not None:
        out["Sentence_No"] = df_raw.iloc[:, sent_col]
    return out


def build_master_metadata(dataset_root: Path, output_path: Path) -> pd.DataFrame:
    excel_files = find_excel_files(dataset_root)
    if not excel_files:
        master = build_metadata_from_wav_names(dataset_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        master.to_excel(output_path, index=False)
        return master

    wav_by_name, wav_by_stem, by_gd_age_sent, by_gd_age, by_gd_sent, by_gd, all_entries = (
        build_wav_indexes(dataset_root)
    )

    frames: list[pd.DataFrame] = []
    for excel in excel_files:
        df = None
        # Some Excel files contain extra blank rows before the header.
        # Try several header offsets and keep the first that yields
        # the expected columns after normalization.
        for header_row in [0, 1, 2, 3, None]:
            try:
                if header_row is None:
                    tmp = pd.read_excel(excel, header=None)
                    # If header is missing entirely, we can't map columns reliably.
                    # Keep for fallback below.
                    df_try = tmp
                else:
                    df_try = pd.read_excel(excel, header=header_row)
            except Exception:
                continue

            df_try = normalize_columns(df_try)
            if {"File_Name", "Gender"}.issubset(set(df_try.columns)):
                df = df_try
                break

        if df is None:
            raw = pd.read_excel(excel, header=None)
            inferred = infer_columns_by_content(raw)
            if inferred is not None and {"File_Name", "Gender"}.issubset(set(inferred.columns)):
                df = inferred
            else:
                df = normalize_columns(pd.read_excel(excel))

        # Ensure required columns exist (may be NaN if not in file).
        df["Source_Excel"] = str(excel)
        # Group prefix (Gxx) is encoded in the parent folder name.
        # Example: .../GROUP_01/Group_01_MetaData.xlsx -> G01
        grp_name = excel.parent.name
        m = re.search(r"(\\d+)", grp_name)
        group_num = int(m.group(1)) if m else None
        df["Group_Code"] = f"G{group_num:02d}" if group_num is not None else ""
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    required = {"File_Name", "Gender"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    # Keep raw gender code (M/F/C or E/K/C) for reconstruction.
    master["Gender_Raw"] = master["Gender"]

    def reconstruct_filename_from_columns(row: pd.Series) -> str:
        # Only attempt reconstruction when File_Name is blank.
        file_name = str(row.get("File_Name", "") or "").strip()
        if file_name and file_name.lower() not in {"nan", "none"}:
            return file_name

        group_code = str(row.get("Group_Code", "") or "").strip()
        subject_id = str(row.get("Subject_ID", "") or "").strip()
        gender_code = str(row.get("Gender_Raw", "") or "").strip()
        age = row.get("Age", None)
        feeling = str(row.get("Feeling", "") or "").strip()
        sentence_no = row.get("Sentence_No", None)

        if not (group_code and subject_id and gender_code and feeling and sentence_no is not None):
            return ""
        try:
            age_int = int(age)
            sent_int = int(sentence_no)
        except Exception:
            return ""

        # Subject_ID often already looks like D01. Keep normalized form for matching.
        subject_norm = normalize_for_match(subject_id)
        # normalize_for_match turns D01 -> D1, which is fine for matching
        return f"{group_code}_{subject_norm}_{gender_code}_{age_int}_{feeling}_{'C' + str(sent_int)}.wav"

    # Some metadata rows may have an empty/NaN 'File name' field.
    # Rebuild missing filenames using other columns.
    master["File_Name"] = master["File_Name"].fillna("").astype(str)
    missing_name_mask = master["File_Name"].str.strip().str.lower().isin({"", "nan", "none"})
    if missing_name_mask.any():
        master.loc[missing_name_mask, "File_Name"] = master.loc[missing_name_mask].apply(
            reconstruct_filename_from_columns, axis=1
        )

    # Remove header rows / invalid rows that don't look like actual wav records.
    # These often show up as 'Dosya_Adi' / 'Cinsiyet' etc.
    master["File_Name"] = master["File_Name"].fillna("").astype(str).str.strip()
    invalid_mask = master["File_Name"].str.lower().isin(
        {"", "nan", "none", "dosya_adi", "dosya adi", "file name", "filename"}
    ) | (
        master["File_Name"].str.contains("dosya", case=False, na=False)
        & ~master["File_Name"].str.contains(".wav", case=False, na=False)
    )
    master = master.loc[~invalid_mask].copy()

    # Convert gender codes to class names for evaluation.
    master["Gender"] = master["Gender_Raw"].apply(
        lambda g: g
        if str(g).strip() in {"Male", "Female", "Child"}
        else parse_gender_from_code(str(g))
    )

    master["File_Name"] = master["File_Name"].fillna("").astype(str).str.strip()
    master["Audio_Path"] = master["File_Name"].apply(
        lambda name: resolve_audio_path(
            str(name),
            wav_by_name,
            wav_by_stem,
            by_gd_age_sent,
            by_gd_age,
            by_gd_sent,
            by_gd,
            all_entries,
        )
        if str(name).strip()
        else ""
    )
    master["Audio_Exists"] = master["Audio_Path"].str.len() > 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_excel(output_path, index=False)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined metadata file from all group Excel files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Dataset"),
        help="Path to dataset root containing group folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/master_metadata.xlsx"),
        help="Output Excel path for merged metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = build_master_metadata(args.dataset_root, args.output)
    total = len(master)
    ok = int(master["Audio_Exists"].sum())
    print(f"Master metadata created: {args.output}")
    print(f"Total rows: {total}")
    print(f"Audio files found: {ok}")
    print(f"Missing audio files: {total - ok}")


if __name__ == "__main__":
    main()
