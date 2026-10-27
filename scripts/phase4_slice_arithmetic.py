#!/usr/bin/env python3
"""Controle arithmetique independant du slicing dense-TP par format (phase 4).

Re-implementation autonome (stdlib uniquement, aucune dependance Rust) du
calcul d'octets utilise par dense_tp_slice_bytes
(crates/hipfire-arch-qwen35/src/qwen35/load.rs) et slice_uniform_group_cols
(crates/hipfire-runtime/src/tp_shard.rs) :

  qt13 (MQ4-G256) / qt44 (MQ4V2-G256) : groupe 256, 136 o par groupe
  qt3  (Q8_0)                         : groupe 32,  34 o par groupe
  row_bytes         = (k / groupe) * octets_par_groupe
  tranche de rang   = par ligne, groupes [start/groupe .. end/groupe)

Jusqu'a la phase 3 comprise, le loader sliceait tout format avec groupe=256 ;
pour qt3 cela est byte-identique si et seulement si k % 256 == 0 (une tranche
de 256 colonnes = 8 groupes qt3 alignes). La phase 4 slice avec le groupe du
format, ce qui debloque les largeurs qt3 « G32 seulement ».

Modes :
  (sans argument)      batterie integree d'identites arithmetiques ; exit 1 si
                       une identite echoue.
  --k K [--k ...]      verifications rapides sur des valeurs de k (qt3 et
                       qt13) : row_bytes, decoupe tp2, equivalence legacy.
  --hfq FICHIER [--tp N]
                       audit d'un artefact HFQ reel : lecture header+index
                       (jamais les blobs), verifications par tenseur 2-D des
                       formats denses (taille de blob vs (m, k, groupe),
                       k % groupe, champ group_size), classification
                       G256-exact / G32-only, et decoupe par groupes entiers
                       equilibree sur tp pour les projections column-parallel
                       (o_proj, down_proj, out_proj). Les layouts de production
                       (tetes d'attention, macro-unites Delta) restent cote
                       Rust (dense_tp_rank_layouts) et sont pinnes par les
                       tests CPU ; ce script ne les re-derive pas.
"""

import argparse
import json
import struct
import sys

FORMATS = {
    13: ("MQ4-G256", 256, 136),
    44: ("MQ4V2-G256", 256, 136),
    3: ("Q8_0", 32, 34),
}
DENSE_TP2_GATE = "dense TP2 requires MQ4-G256 qt13, MQ4V2 qt44 or Q8_0 qt3"
COLS_SUFFIXES = (".o_proj.weight", ".down_proj.weight", "linear_attn.out_proj.weight")


def balanced_range(rank, parts, total):
    """Meme formule que ShardConfig::balanced_range (tp_shard.rs)."""
    base, rem = divmod(total, parts)
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def row_bytes(k, group, group_bytes):
    if k % group:
        raise ValueError(f"k={k} not a multiple of group={group}")
    return (k // group) * group_bytes


def cols_slice_span(m, k, start, end, group, group_bytes, blob_bytes):
    """Renvoie (offset_par_ligne, octets_par_ligne) de la tranche [start, end)."""
    if (
        m == 0
        or group == 0
        or start > end
        or end > k
        or k % group
        or start % group
        or end % group
        or blob_bytes % m
    ):
        raise ValueError(
            f"invalid uniform-group column slice {start}..{end} "
            f"for {m}x{k}, group={group}, bytes={blob_bytes}"
        )
    groups = k // group
    row = blob_bytes // m
    if row % groups:
        raise ValueError(f"row bytes {row} are not divisible by {groups} quant groups")
    per_group = row // groups
    if per_group != group_bytes:
        raise ValueError(
            f"blob does not match the format: per-group {per_group} != {group_bytes}"
        )
    first = start // group
    span = (end - start) // group * per_group
    return first * per_group, span


def legacy_group256_span(m, k, start, end, group_bytes, blob_bytes):
    return cols_slice_span(m, k, start, end, 256, 8 * group_bytes, blob_bytes)


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS {name}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as exc:
        print(f"  FAIL {name} — {exc}")
        return False


def battery():
    ok = True
    print("== identites arithmetiques (batterie integree) ==")

    def qt3_k512_tp2():
        m, k, gb = 256, 512, 34
        blob = m * (k // 32) * gb
        off0, span0 = cols_slice_span(m, k, 0, 256, 32, gb, blob)
        off1, span1 = cols_slice_span(m, k, 256, 512, 32, gb, blob)
        loff0, lspan0 = legacy_group256_span(m, k, 0, 256, gb, blob)
        loff1, lspan1 = legacy_group256_span(m, k, 256, 512, gb, blob)
        assert (off0, span0) == (loff0, lspan0), "rang 0 qt3 != legacy G256"
        assert (off1, span1) == (loff1, lspan1), "rang 1 qt3 != legacy G256"
        assert (span0, span1) == (272, 272), "spans attendus 272 o/ligne"
        return f"row=544 o, rangs 272 o/ligne, equivalent legacy G256"

    ok &= check("qt3 k=512 tp2 : equivalent legacy G256 (artefact cible)", qt3_k512_tp2)

    def qt3_k96_tp2():
        m, k, gb = 8, 96, 34
        blob = m * (k // 32) * gb
        off0, span0 = cols_slice_span(m, k, 0, 64, 32, gb, blob)
        off1, span1 = cols_slice_span(m, k, 64, 96, 32, gb, blob)
        assert (off0, span0) == (0, 68), "rang 0 attendu 68 o/ligne (2 groupes)"
        assert (off1, span1) == (68, 34), "rang 1 attendu 34 o/ligne (1 groupe)"
        try:
            legacy_group256_span(m, k, 0, 64, gb, blob)
        except ValueError:
            return "G32-only : legacy G256 refuse (k%256!=0), phase 4 admet"
        raise AssertionError("legacy G256 n'aurait pas du admettre k=96")

    ok &= check("qt3 k=96 tp2 : largeur hors G256 admise, legacy refusee", qt3_k96_tp2)

    def qt13_k512_tp2():
        m, k, gb = 256, 512, 136
        blob = m * (k // 256) * gb
        off0, span0 = cols_slice_span(m, k, 0, 256, 256, gb, blob)
        off1, span1 = cols_slice_span(m, k, 256, 512, 256, gb, blob)
        assert (span0, span1) == (136, 136), "spans attendus 136 o/ligne"
        return "row=272 o, rangs 136 o/ligne (1 groupe G256 par rang)"

    ok &= check("qt13 k=512 tp2 : groupe 256 inchange", qt13_k512_tp2)

    def refus_qt3():
        m, k, gb = 8, 96, 34
        blob = m * (k // 32) * gb
        for start, end in ((0, 48), (48, 96)):
            try:
                cols_slice_span(m, k, start, end, 32, gb, blob)
            except ValueError:
                continue
            raise AssertionError(f"{start}..{end} n'aurait pas du passer (fin non G32)")
        try:
            cols_slice_span(m, 50, 0, 32, 32, gb, blob)
        except ValueError:
            return "fin non alignee et k non multiple refuses"
        raise AssertionError("k=50 n'aurait pas du passer")

    ok &= check("qt3 : refus fail-closed (fin non G32, k non multiple)", refus_qt3)

    def refus_qt13():
        m, k, gb = 4, 512, 136
        blob = m * (k // 256) * gb
        try:
            cols_slice_span(m, k, 0, 128, 256, gb, blob)
        except ValueError:
            return "tranche 0..128 hors G256 refusee (message inchange)"
        raise AssertionError("0..128 n'aurait pas du passer en G256")

    ok &= check("qt13 : refus fail-closed inchange (fin non G256)", refus_qt13)

    def blob_sizes():
        for qt, (_, group, gb) in FORMATS.items():
            for k in (32, 96, 256, 512, 1024, 5120):
                if k % group:
                    continue
                m = 7
                assert row_bytes(k, group, gb) == (k // group) * gb
                assert m * row_bytes(k, group, gb) == m * (k // group) * gb
        return "row_bytes = (k/groupe)*octets_groupe pour tous les (qt, k) admis"

    ok &= check("tailles de blobs : identite (m, k, groupe)", blob_sizes)
    return ok


def read_hfq_index(path):
    with open(path, "rb") as fh:
        head = fh.read(32)
    if len(head) < 32 or head[:4] != b"HFQM":
        raise ValueError(f"{path}: bad HFQ magic")
    version, arch_id, n_tensors = struct.unpack_from("<III", head, 4)
    metadata_offset, data_offset = struct.unpack_from("<QQ", head, 16)
    with open(path, "rb") as fh:
        buf = fh.read(min(64 * 1024 * 1024, data_offset + (1 << 20)))
    metadata, index_off = _scan_metadata(buf, metadata_offset)
    tensors = []
    (count,) = struct.unpack_from("<I", buf, index_off)
    pos = index_off + 4
    if count != n_tensors:
        raise ValueError(f"index count {count} != header n_tensors {n_tensors}")
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", buf, pos)
        pos += 2
        name = buf[pos : pos + name_len].decode("utf-8")
        pos += name_len
        qt, n_dims = struct.unpack_from("<BB", buf, pos)
        pos += 2
        shape = list(struct.unpack_from(f"<{n_dims}I", buf, pos))
        pos += 4 * n_dims
        (group_size,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        (data_size,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        tensors.append((name, qt, shape, group_size, data_size))
    return version, arch_id, json.loads(metadata), tensors


def _scan_metadata(buf, off):
    depth = 0
    i = off
    in_str = False
    esc = False
    while i < len(buf):
        c = buf[i]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                return bytes(buf[off : i + 1]), i + 1
        i += 1
    raise ValueError("unterminated metadata JSON")


def audit_hfq(path, tp):
    version, arch_id, metadata, tensors = read_hfq_index(path)
    config = metadata.get("config", {})
    print(f"== audit HFQ : {path}")
    print(
        f"   version={version} arch_id={arch_id} tenseurs={len(tensors)} "
        f"hidden_size={config.get('hidden_size')} intermediate_size={config.get('intermediate_size')}"
    )
    failures = 0
    audited = 0
    g32_only = 0
    for name, qt, shape, group_field, data_size in tensors:
        if qt not in FORMATS or len(shape) != 2:
            continue
        audited += 1
        label, group, gb = FORMATS[qt]
        m, k = shape
        kind = "cols" if name.endswith(COLS_SUFFIXES) else "rows"
        problems = []
        split_note = ""
        if k % group:
            problems.append(f"k={k} non multiple de {group} (refus preflight)")
        else:
            expected = m * (k // group) * gb
            if data_size != expected:
                problems.append(f"blob {data_size} != attendu {expected}")
            elif kind == "cols":
                n_groups = k // group
                if n_groups < tp:
                    split_note = (
                        f" | decoupe tp{tp} impossible ({n_groups} groupes < {tp})"
                    )
                    problems.append(f"n_groups {n_groups} < tp {tp}")
                else:
                    spans = []
                    total = 0
                    for rank in range(tp):
                        start, end = balanced_range(rank, tp, n_groups)
                        rank_bytes = (end - start) * gb * m
                        spans.append(f"{rank_bytes}o")
                        total += rank_bytes
                    if total != data_size:
                        problems.append(f"somme rangs {total} != blob {data_size}")
                    else:
                        split_note = f" | tp{tp}: {'/'.join(spans)} par rang"
        want_group = 32 if qt == 3 else 256
        if group_field != want_group:
            problems.append(f"group_size={group_field} != {want_group}")
        status = "OK " if not problems else "FAIL"
        failures += bool(problems)
        cls = "G256-exact" if k % 256 == 0 else "G32-only"
        g32_only += cls == "G32-only"
        line = (
            f"   {status} [{label:>10}] {kind} {name}: [{m}x{k}] "
            f"blob={data_size} {cls}{split_note}"
        )
        if problems:
            line += " | " + "; ".join(problems)
        print(line)
    print(
        f"   audit : {audited} tenseurs denses verifies, "
        f"{g32_only} hors G256 (capacite phase 4), {failures} echec(s)"
    )
    return failures == 0


def quick_ks(ks):
    ok = True
    print("== verifications rapides (--k) ==")
    for k in ks:
        for qt, (label, group, gb) in FORMATS.items():
            if k % group:
                print(f"   SKIP k={k} {label}: k % {group} != 0 (refuse)")
                continue
            m = 4
            rb = row_bytes(k, group, gb)
            n_groups = k // group
            spans = []
            for rank in range(2):
                start, end = balanced_range(rank, 2, n_groups)
                spans.append(f"{(end - start) * gb}o")
            equiv = ""
            if qt == 3 and k % 256 == 0:
                equiv = ", equivalent legacy G256"
            print(
                f"   k={k} {label}: row={rb}o, {n_groups} groupes, "
                f"tp2 {'/'.join(spans)} par rang{equiv}"
            )
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, action="append", help="valeur de k a verifier")
    parser.add_argument("--hfq", help="artefact HFQ a auditer")
    parser.add_argument("--tp", type=int, default=2, help="topologie pour la decoupe (defaut 2)")
    args = parser.parse_args()
    ok = True
    if args.k:
        ok &= quick_ks(args.k)
    if args.hfq:
        ok &= audit_hfq(args.hfq, args.tp)
    if not args.k and not args.hfq:
        ok &= battery()
    if not ok:
        print("RESULTAT : ECHEC")
        return 1
    print("RESULTAT : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
