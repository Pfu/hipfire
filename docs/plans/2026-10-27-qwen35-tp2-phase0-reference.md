# Phase 0 — compte rendu de reference (qwen35 tp2 q8)

**Date :** 2026-10-27 (run `20261027T060706Z`, produit par `scripts/phase0_reference.py`)
**Binaire :** hipfire 0.4.3+patch.1 (`af4cb2f62fc4`, branche develop)
**Machine :** 2x gfx1100 24 GiB, PCIe 28.3 GB/s (calibre 2026-10-26) ; RAM 123.4 GiB (109.5 dispo)
**Verrou :** `flock -w 3600 /tmp/hipfire-gpu.lock` ; aucune commande GPU hors verrou

## Artefacts scelles (sha256)

| role | fichier | taille | sha256 |
|---|---|---|---|
| trunk q8 (cible) | `/home/spf/qwen3.8-27b.q8.hfq` | 28 592 024 576 o (26.63 GiB) | `7c1ad7a49830248509d72ad5a62101a65062e7228bcc545079d8f7c8755d27d7` |
| reference mq4 | `/home/spf/.hipfire/models/qwen3.8-27b.mq4-xts` | 14 987 185 152 o (13.96 GiB) | `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae` |
| tete MTP | `/home/spf/.hipfire/models/qwen3.8-27b.mtp` | 225 716 224 o (0.21 GiB) | `f0d46d07ded75abc095ebfdbccc167c68418a6515b187bf810395faa460bd32e` |

Aucun telechargement (artefacts deja presents). HF_TOKEN presente dans l'environnement (jamais
affichee). Fichiers partiels de pulls anterieurs listes dans le rapport brut (non geres ici).

## Peer access : OUI

```
[loader] dense qwen TP load complete: 2 ranks, peer_access=true
```

La voie TP dense est viable sur cette paire de cartes ; la question chere de la phase 0 est
tranchee en faveur de tp2.

## Attache MTP en tp2 (mq4-xts, `--spec mtp`)

```
MTP head loaded (sidecar .../qwen3.8-27b.mtp, dense TP rank 0): n_embd=5120 vocab=248320
qwen35 MTP speculator enabled (compressed-serial, K=3)
qwen35 MTP prompt fill route: ar (AR prefill route)
[req run] drafter=mtp tau=2.00 tok/s=68.3 decode (55 tok, 18 windows)
```

La route cible « tp2 + MTP » fonctionne des aujourd'hui sur l'artefact mq4 : 68.3 tok/s en decode
MTP contre 51.9 tok/s AR (meme artefact, meme topologie), soit +31.6 %.

## Baselines de debit (AR pur, `--spec off`, tg128, 3 processus frais, 5 runs chacun)

| topologie | decode tok/s (pool, 15 runs) | prefill tok/s (pool) |
|---|---|---|
| mq4 **tp2** | 51.91 (min 51.7, max 52.0) | 379.26 |
| mq4 **pp2** | 44.71 (min 44.7, max 44.8) | 344.31 |

- tp2 domine pp2 de **+16.1 %** en decode sur cette machine.
- **Reserve sur le prefill** : prompt de 24 tokens (defaut du bench) — ces nombres mesurent
  l'overhead de lancement, pas le debit de prefill (warning du bench). Un tok/s de prefill reel
  demande `--bench-prompt-file` avec >= 256 tokens ; a refaire avant d'en faire une reference.
- Contexte des mesures : `memory.max_seq` 32768 (config).

## Refus de cadrage (premier passage, conserves au journal du plan)

- draft DFlash resolu automatiquement (sidecar declare des tiers qwen3.8:27b) :
  `run tp2` sans spec -> `EP serving (tp>1) does not support DFlash drafters in v1; reload
  without a draft.` ; `bench pp2` -> `DFlash speculative decode requires pp=1 in v1 (...)`.
- `HIPFIRE_PP_DFLASH=1` reste ecarte (precedent du plan : load accepte, generation sans
  spec-decode cross-card). Les mesures epinglent donc `--spec off` / `--spec mtp`.

## Caveat de mesure corrige au passage

Le premier passage donnait `fit --tp 2 -> DOES NOT FIT` : le fit s'executait apres les benchs et
voyait ~14 GiB libres (daemon du bench encore resident pendant son idle timeout). Le script
execute maintenant le fit avant les mesures ; le verdict fit de cette passe est `does-not-fit`
pour la meme raison d'ordre — la passe suivante le montrera sur cartes propres.

## Porte de sortie

Peer access oui/non : **oui**. Artefacts : scelles (table ci-dessus). tok/s de reference :
tp2 51.91 / pp2 44.71 decode AR (prefill a reprendre avec un vrai prompt). Prompt et mode :
`Hello`, greedy (`--temp 0`). **La phase 0 est close ; les phases 1 (spike kernels Q8) et 3
(preflight qt3) sont debloquees.**
