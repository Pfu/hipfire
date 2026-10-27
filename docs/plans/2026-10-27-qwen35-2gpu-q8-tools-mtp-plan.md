# Qwen3.5 (arch 5) — 2 GPU + tool_call + MTP en Q8 : plan d'implementation ordonne

**Status:** actif — phases 0, 2 et 3 closes sur materiel (journal et compte rendu plus bas) ; phase 4 close sur materiel (slicing par groupe de format : tests CPU verts, chargement tp2 complet sans erreur de slice) ; phases 4-5 re-calibrees par le run de cloture de la phase 3 (slicing et admission kernels deja fonctionnels sur l'artefact cible G256 ; restent le prefill long >= 256 tokens et la parite tp2-vs-mono) ; phase 1 (spike) ouverte.
**Auteur:** (agent)
**Cree:** 2026-10-27
**Specification:** `docs/plans/2026-10-27-qwen35-2gpu-q8-tools-mtp-spec.md` (absente du depot
au 2026-10-27 : a creer ou rapatrier avant la phase 3)

Ce plan est **ordonne par dependance**, pas par priorite percue : la phase 1 (spike) decide si la
phase 5 coute peu ou beaucoup, et la phase 2 est la seule entierement parallelisable.

```
Phase 0  cadrage + mesures de reference        (aucun code)
Phase 1  spike : formes locales Q8             (aucun code de production)
Phase 2  hipfire fit --tp                      (independant, parallelisable)
Phase 3  preflight qt3 + tests CPU             (depend de 0)
Phase 4  slicing qt3 + lecture par rang        (depend de 3)
Phase 5  admission kernels Q8 decode puis prefill  (depend de 1 et 4)
Phase 6  graphe TP2 + attache MTP sur trunk Q8 (depend de 5)
Phase 7  validation bout en bout + receipt/docs (depend de 6, 2)
Phase 8  SPEC-1 : outils sur PP                (independant ; apres 7 dans l'ordre conseille)
Phase 9  SPEC-3 : MTP sur mesh PP              (differe ; depend de 8)
```

Regles applicables a toutes les phases :

- Une phase = une branche = un (ou plusieurs) commit(s) decrivant les fichiers changes. Jamais de
  commit sur `master`.
- Aucune phase ne change un defaut : le support qt3 en TP dense est une **capacite nouvelle**, pas
  un basculement pour un chemin qui marche (MQ4).
- Fail-closed : un format ou une forme non admis est refuse en nommant le tenseur, le format et le
  mode. Aucun fallback silencieux.
- Les messages de refus existants (MQ4, qt53, geometrie `pp`/`tp`) gardent leur texte : ils sont
  pinnes par des tests CPU et par `docs/load-refusal-wording.md`.
- Aucune phase n'est fusionnee dans une autre pour aller plus vite : la porte de sortie d'une phase
  est ce qui autorise la suivante.

---

## Phase 0 — Cadrage et mesures de reference (aucun code)

**But.** Verrouiller l'environnement de mesure et le point de depart, avant de toucher au code.

**Travaux.**
1. Confirmer le peer access entre les deux gfx1100 (le TP dense reduit « over peer access »,
   `crates/hipfire-runtime/src/multi_gpu.rs`) : le load logue
   `dense qwen TP load complete: N ranks, peer_access=...`.
2. Fixer les artefacts : trunk q8, tete MTP (trailer `.mq4-mtp`, sidecar `.mtp`, ou frere du
   trunk) et l'artefact mq4 equivalent qui sert de reference de non-regression.
3. Mesurer la reference actuelle : `pp2` (ce qui marche) et, si une tier mq4 existe, `tp2` en
   mq4 — c'est la seule ligne de base « outils + MTP » disponible avant les phases suivantes.
4. Noter les commandes GPU : toute commande GPU est prise sous verrou, p.ex.
   `flock -w 3600 /tmp/hipfire-gpu.lock hipfire bench ...`.

**Porte de sortie.** Un compte rendu horodate : peer access oui/non, chemins des artefacts,
`tok/s` de reference, prompt et mode greedy.

**Interdit.** Modifier du code.

---

## Phase 1 — Spike : admission des formes locales Q8 (aucun code de production)

**But.** Repondre a la question la plus chere avant d'ecrire quoi que ce soit : les kernels Q8_0
admettent-ils les largeurs **shardees** du TP dense, en decode et surtout en prefill ?

**Prerequis.** Phase 0.

**Travaux.** Harnais lab (pas de code produit, pas de nouveau defaut) qui, pour chaque op du
forward dense TP et pour la ou les formes locales reelles (`o_in = n_heads*head_dim/2`,
`n_kv/2`, `value_dim/2`, `hidden/2`, `K` correspondants), essaie le kernel Q8_0 de la mono-carte
et rapporte admis / refuse avec le message exact. Ops a couvrir : attention q/k/v/o, GDN
`in_proj_qkv/z/a/b` + `out_proj`, FFN gate/up/down, `output` (Gemv), en **decode** (M=1) et en
**prefill** (tuiles WMMA, plusieurs hauteurs de chunk).

**Porte de sortie.** Une liste exhaustive `(op, M, K, admis/refuse, message)`. Elle determine le
contenu de la phase 5 :
- tout admis => phase 5 = verification + tests, pas de nouveau kernel ;
- formes refusees => phase 5 inclut leur admission (regle ou fallback explicite), et l'estimation
  de la spec est revisee avant de continuer.

**Interdit.** Livrer un chemin de production sur la base d'un spike non concluant.

---

## Phase 2 — `hipfire fit --tp` (independant, parallelisable)

**But.** Ne plus afficher un plan mono-carte pour un load multi-GPU ; c'est exactement la confusion
qui a masque le refus de format.

**Prerequis.** Aucun. Branche independante.

**Travaux.** `FitArgs` (`crates/hipfire-cli/src/fit/mod.rs:31`) expose `--tp` (et, si utile, `--pp`)
alors que `admission_inputs` lit deja `params["tp"]` (`fit/mod.rs:1268`) et que `model_params`
utilise le meme mapper que `run`/`serve`. Le plan doit afficher la topologie effective et, quand un
`tp > 1` est demande, dire explicitement que les poids par rang ne sont pas projetes (ou les
projeter).

**Porte de sortie.** `hipfire fit <modele> --tp 2` affiche `topology tp2` et ne pretend plus que le
modele ne tient pas sur une carte quand le load demande en tient deux.

**Note.** Cette phase ne debloque rien de fonctionnel : elle rend le diagnostic honnete.

---

## Phase 3 — Preflight qt3 + tests CPU

**Prerequis.** Phase 0 (et l'arbitrage de phase 1 s'il conclut a un refus structurel).

**Travaux.**
1. Generaliser le validateur de projection en validateur par format (MQ4 : groupe 256,
   `(k/256)*136` ; Q8_0 : groupe 32, `(k/32)*34`, `k % 32 == 0`), avec messages MQ4 inchanges.
   Fichier : `crates/hipfire-arch-qwen35/src/qwen35/load.rs` (`validate_mq4_proj_info`,
   `expected_mq4_bytes`, `is_dense_proj_qt`, appels de `preflight_weights_dense_tp`).
2. Accepter qt3 sur les projections listees, continuer a refuser qt53 et les deviations de
   geometrie.
3. Tests CPU : qt3 accepte (geometrie correcte), qt3 refuse si `k % 32 != 0` ou groupe != 32,
   qt53 toujours refuse, messages MQ4 inchanges au caractere pres.

**Porte de sortie.** Tests CPU verts ; un fichier q8 ne s'arrete plus sur le message de format (il
peut encore echouer plus loin, c'est l'objet des phases suivantes).

---

## Phase 4 — Slicing qt3 et lecture par rang

**Prerequis.** Phase 3.

**Travaux.** Dans `load_weight_tensor_dense_tp` (`load.rs:4354`) : `DenseTpSlice::Cols` decoupe avec
le groupe du format (32 pour qt3) ; `Rows`/`RowsMulti` inchanges ; reutiliser
`load_weight_tensor_raw` (qt3 => `DType::Q8_0`, `load.rs:973-990`). Test unitaire de l'invariant
« largeurs locales alignees » sur les layouts (`config.rs:192`).

**Porte de sortie.** Un chargement atteint l'upload par rang sans erreur de slice ; les tests de
layout passent en CPU.

---

## Phase 5 — Admission des kernels Q8, decode puis prefill

**Prerequis.** Phase 1 (liste) et phase 4.

**Travaux.** Dans l'ordre :
1. **Decode** : `forward_scratch_dense_tp*` (`forward.rs:3680-3765`) — faire admettre chaque op Q8
   aux formes locales ; tests ciblés par op.
2. **Prefill** : `forward_prefill_dense_tp` — memes ops, formes de tuiles ; si le spike a identifie
   des refus, les traiter ici (extension de regle d'admission, ou fallback explicite et documente).
3. Aucun changement pour MQ4 : verifier par test que les chemins MQ4 ne bougent pas.

**Porte de sortie.** Un run decode et un run prefill complets sur un trunk q8 en `--tp 2`, sans
erreur d'admission, sur formes reelles.

---

## Phase 6 — Graphe TP2 et attache MTP sur trunk Q8

**Prerequis.** Phase 5.

**Travaux.**
1. Verifier la capture whole-token TP2 gfx1100/gfx1201 (`crates/hipfire-runtime/src/multi_gpu.rs:1464-1473`)
   avec un trunk Q8 : chaque kernel du segment capture doit etre admis, sinon la capture est
   desactivee pour ce codec (jamais un resultat faux).
2. Verifier l'attache MTP : `load_model_tp_qwen35_dense` (`lib.rs:5380-5415`),
   `Qwen35DenseTpGuard` (`spec_build.rs:148-230`), `mtp_dense_tp` — trunk Q8 + `--spec mtp`.
   `auto` doit continuer a rester AR (aucun changement de defaut).

**Porte de sortie.** Un decode MTP fonctionne sur trunk Q8 en tp2, avec le log d'attache attendu ;
la voie AR reste intacte.

---

## Phase 7 — Validation bout en bout, docs, receipt

**Prerequis.** Phase 6 et phase 2.

**Travaux.**
1. Parite TP2-vs-mono-carte pour un artefact qt3 en reutilisant le critere existant de
   `crates/hipfire-arch-qwen35/tests/route_oracle_mesh.rs` (« tokens exacts, |delta logit| <= 1.0 »),
   commande sous verrou (`flock`), 3 processus frais.
2. Bout en bout serveur OpenAI-compatible : `hipfire serve <trunk q8> --tp 2 --spec mtp` ;
   requete `tools` => `finish_reason: "tool_calls"` ; `stop` honore ; check `max_seq`.
3. Non-regression mq4 tp2 (meme harnais, artefact mq4).
4. Docs : `docs/multi-gpu.md` (section TP dense : formats admis), `docs/CONFIG.md` si un message de
   refus est cite, `docs/speculation-support-inventory.md` (MTP sur mesh dense), `CHANGELOG.md`.
5. Receipt de debit (decode et prefill) avec date, binaire et identite du modele.

**Porte de sortie.** Critere d'acceptation de la spec section 5 satisfait, docs a jour, receipt
ecrit. C'est ici que la cible « 2 GPU + outils + MTP en Q8 » est declaree atteinte.

---

## Phase 8 — SPEC-1 : outils sur la route pipeline-parallel

**Prerequis.** Aucun (techniquement independant). **Ordre conseille : apres la phase 7**, pour ne
pas elargir la surface de changement avant que la cible soit acquise.

**Travaux.** `supports_tools()` (`ar.rs:947`) + `PipelineParallel` ; mise a jour de
`crates/hipfire-generate/tests/generation_route_matrix_tests.rs` (`SAFE_ROUTES`,
`exact_safe_set_is_...`, compteur 11 -> 12, assertion negative) ; **preuve de parite de fil** entre
la mono-carte et `generate_multi` (mêmes evenements, meme terminal, meme traitement d'un appel
malforme) ; CHANGELOG et docs de routes.

**Porte de sortie.** Test de parite de fil vert ; une requete `tools` fonctionne sur un load `pp2`.
**Hors perimetre** : `stop`, MTP, VMM KV sur PP.

**Risque explicite.** La whitelist est celle des producteurs « proven semantic-safe » : si la
parite de fil ne peut pas etre demontree, la fonctionnalite est abandonnee.

---

## Phase 9 — SPEC-3 : MTP natif sur le mesh pipeline-parallel (differe)

**Prerequis.** Phase 8. **A ne pas demarrer sans besoin explicite** : la phase 7 couvre deja la
cible, et le precedent `HIPFIRE_PP_DFLASH=1` (load accepte, generation sans spec-decode cross-card)
illustre le cout d'un chemin annonce mais inexploitable.

**Travaux (esquisse).** Tete MTP sur la bande de sortie ; draft sur la derniere bande ; verify a
travers les bandes avec la tuyauterie existante ; cap de largeur ; etat KV/DeltaNet par bande ;
admission fail-closed ; validation token-exacte PP-MTP vs mono-carte MTP.

**Porte de sortie.** Un spec-decode MTP reel sur `pp > 1`, token-exact, avec un receipt.

---

## Ce qui doit etre tenu pendant tout le plan

- **Pas de defaut modifie** : qt3 en tp est une capacite, `mq4` reste la voie par defaut.
- **Pas de message de refus reformule** sans mise a jour de `docs/load-refusal-wording.md` et des
  tests qui les pinnent.
- **Un refus reste un refus** : jamais de downgrade silencieux de format, de KV ou de topologie.
- **Chaque phase est independamment annulable** : aucune ne doit laisser le depot dans un etat ou un
  load qui marchait avant echoue apres.

---

## Journal d'execution

- **2026-10-27 — Phase 4 livree (slicing par groupe de format ; validation CPU/GPU en attente)** :
  - `dense_tp_slice_bytes` (nouveau, pur, `crates/hipfire-arch-qwen35/src/qwen35/load.rs`) :
    extraction de la selection de tranche de `load_weight_tensor_dense_tp` — `Rows`/`RowsMulti`
    inchanges au caractere pres, `Cols` decoupe avec le groupe du format (`DenseProjFormat.group` :
    256 pour qt13/qt44, 32 pour qt3) au lieu du 256 en dur ; `load_weight_tensor_raw` reste le seul
    point d'upload (qt3 => `DType::Q8_0`). La porte de format reste la premiere verification
    (`dense TP2 requires MQ4-G256 qt13, MQ4V2 qt44 or Q8_0 qt3` verbatim) ; `is_dense_proj_qt`
    supprime (absorbe par le match sur `dense_proj_format`, source de verite unique). Fail-closed
    inchange : une tranche non alignee au groupe du format est refusee par
    `slice_uniform_group_cols` (message runtime verbatim), aucun fallback silencieux.
  - Equivalence legacy conservee : pour `k % 256 == 0`, la decoupe G32 selectionne exactement les
    memes octets que l'ancienne G256 en dur (272 o/ligne/rang sur k=512) — pinnee par test
    (comparaison directe avec `slice_uniform_group_cols(..., 256)`). Nouvelle capacite : les
    largeurs qt3 « G32 seulement » (ex. k=96) sont desormais sliceables ; hors de ces largeurs,
    refus explicite (meme message runtime que la phase 3).
  - Tests CPU (`mod dense_tp_slice_group_tests`, fixtures HFQ via `write_hfqm_package_mem`, style
    du mod phase 3 dont il reutilise les helpers publies `pub(super)`) : qt3 `o_proj` [256x512]
    slice par rang a G32 (octets exacts par rang + equivalence legacy), qt3 k=96 hors G256,
    qt13/qt44 inchanges a G256, `Rows`/`RowsMulti` inchanges sur les deux familles, refus
    fail-closed verbatim (fin non G32, k non multiple de 32, blob tronque, row_bytes non divisibles
    par les groupes, fin non G256 qt13), invariant « largeurs locales alignees » des layouts
    (`dense_tp_rank_layouts`) verifie par format (256 et 32) pour les trois formats admis : borne
    start/end des ranges `Cols` du loader (q x head_dim, dv x value_dim, ffn) sur la geometrie
    fixture (tp2) et une geometrie sonde tp2..4 (chemin GQA splitté) ; tp3 sur la geometrie
    fixture refuse (2 macro-unites Delta < 3).
  - Script `scripts/phase4_slice_arithmetic.py` : re-implementation autonome de l'arithmetique de
    slicing (batterie integree, `--k`, `--hfq` pour auditer un artefact reel depuis header+index :
    taille de blob vs (m, k, groupe), champ group_size, classification G256-exact / G32-only,
    decoupe tp par groupes entiers pour o_proj/down_proj/out_proj). Cote prod :
    `python3 scripts/phase4_slice_arithmetic.py --hfq ~/qwen3.8-27b.q8.hfq`.
  - Reste a valider : `cargo test -p hipfire-arch-qwen35` (cote utilisateur), puis GPU : le run tp2
    du trunk q8 doit rester identique (artefact tout-G256 : slicing G32 equivalent a G256, verifie
    par test et par le script) ; prefill long (>= 256 tokens, chunks WMMA reels) et parite
    tp2-vs-mono restent l'objet des phases 5 et 7.
  - **2026-10-27 — Tests CPU verts sur materiel** (runs `results/run-20261027T091026Z` — filtre
    `dense_tp` : 28 passed / 0 failed, 435 filtered — et `results/run-20261027T091207Z` — lib
    complete : 438 passed / 0 failed / 25 ignored, les memes seams GPU qu'au run de la porte qt3 ;
    commit `cbf3cdf78`, rustc/cargo 1.101.0-nightly) : compilation sans warning ni erreur (la
    suppression de `is_dense_proj_qt` ne laisse aucun code mort). Porte de sortie CPU atteinte.
  - **2026-10-27 — Phase 4 CLOSE : chargement par rang sans erreur de slice** (run tp2 du trunk q8
    sous le verrou du plan, daemon isole via `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon` ; le
    canari confirme le daemon du build local a jour — l'installation `~/.hipfire/bin/daemon` reste
    perimee, remede durable `hipfire update` toujours en attente) : load complet (`dense qwen TP
    load complete: 2 ranks, peer_access=true`, tape whole-token 128 barriers, KV Q8 VMM par rang,
    51.88 GB lus), prefill + decode 54 tok @ 43.9 tok/s (identique a la cloture precedente —
    attendu : slicing G32 equivalent a G256 sur artefact tout-G256, pinne par test et par le
    script), sortie textuelle coherente. Aucun fallback silencieux, aucun refus.

- **2026-10-27 — Phase 3 livree (code + tests CPU ; validation CPU/GPU en attente)** :
  - `validate_mq4_proj_info` -> `validate_dense_proj_info` (`crates/hipfire-arch-qwen35/src/qwen35/load.rs`) :
    validateur de projection par format — MQ4 inchange (groupe 256, `(k/256)*136`, messages
    `group_size must be 256` / `shape mismatch` / `K={k} not multiple of 256` / `blob length
    mismatch` verbatim), Q8_0 admis (qt=3, groupe 32, `(k/32)*34`, `k % 32 == 0`, messages
    `Q8 group_size must be 32` / `K={k} not multiple of 32`). `expected_mq4_bytes` reste
    (wrapper de la nouvelle `expected_grouped_bytes` ; `validate_embedding_info` inchange) ;
    `is_dense_proj_qt` delegue a `dense_proj_format` — qt3 passe les deux portes de format
    (preflight et `load_weight_tensor_dense_tp`, dont le message cite maintenant les trois
    formats admis ; textes de porte absents de `docs/load-refusal-wording.md`, verifie par grep).
  - Alignements geometriques du preflight (`K=... not G... aligned`, `o_in`, `hidden`)
    parametres par le groupe du format des tenseurs concernes : texte MQ4 identique au
    caractere pres (`not G256 aligned`), Q8_0 exige 32.
  - Fail-closed : qt53 refuse aux deux niveaux (message `reject_hfq_qt53_for_mode` verbatim,
    plus le refus du validateur direct), deviations de geometrie refusees, aucun fallback
    silencieux. Le slicing par rang reste groupe-256 en dur (phase 4) : un q8 peut passer la
    porte de format puis echouer au slice — refuse explicitement par `slice_uniform_group_cols`.
  - Tests CPU purs (`mod dense_tp_preflight_format_tests`, fixtures HFQ minuscules ecrites avec
    `write_hfqm_package_mem` du runtime, geometrie compatible `dense_tp_rank_layouts` tp2 :
    hidden 512, 8 Q heads, Delta 4x4 tetes value_dim 128, 1 couche linear_attention + 1
    full_attention) : qt3 accepte (embed+lm_head qt3), qt13/qt44 acceptent, qt3 refuse
    `k % 32 != 0` et `group_size != 32`, qt53 refuse (preflight et validateur direct), messages
    MQ4 pinnes (group_size, shape, blob, K multiple de 256), porte de format nommant les trois
    formats. Doc de l'exemple `bench_dflash_verify_shapes.rs` mis a jour vers le symbole renomme.
  - Reste a valider : `cargo test -p hipfire-arch-qwen35` (cote utilisateur), puis sur GPU que
    le q8 passe la porte de format (peut encore echouer plus loin — slicing phase 4, admission
    kernels phase 5).
  - **2026-10-27 — Tests CPU verts sur materiel** (run `results/run-20261027T080657Z`,
    commit `11a83236e`, rustc/cargo 1.101.0-nightly) : compilation OK ; lib 432 passed /
    0 failed / 25 ignored (les 25 = fault-seams GPU et suites `#[ignore]` attendues) ;
    mod `dense_tp_preflight_format_tests` 12/12 ok ; suites d'integration `tests/*.rs`
    ok (elles n'avaient pas tourne au run precedent : cargo s'arrete au premier echec) ;
    skips runtime 0 (tous les tests GPU du crate sont `#[ignore]` ou passent). Deux
    correctifs de fixtures ont ete necessaires (norme finale `norm.weight` omise ;
    suffixe ` (hipError=0)` du message HipError dans l'attente qt53). Porte de sortie CPU
    de la phase atteinte ; reste la validation GPU : `run` du trunk q8 en tp2 doit
    depasser le message de format observe en phase 2.
  - **2026-10-27 — Phase 3 CLOSE : porte qt3 passee sur GPU, decode complet** (run manuel
    sous le verrou du plan, daemon isole via `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon` ;
    l'installation `~/.hipfire/bin/daemon` etait perimee — vieille porte de format — et le
    canari du pilote de tests l'a collationnee) :
    `run ~/qwen3.8-27b.q8.hfq --tp 2 --max-seq 4096 --temp 0 --spec off "Hello"` —
    `dense qwen TP load complete: 2 ranks, peer_access=true` ; tape whole-token TP2 (128
    barriers) ; KV Q8 VMM par rang ; tous les kernels q8 admis (decode et prefill :
    embedding/gemm_qkvza/gated_delta_net/gemm_q8_0_residual/gate_up/gemm_qkv/attention/gemv) ;
    prefill + decode 54 tokens @ 43.8 tok/s, sortie textuelle coherente. Aucun fallback
    silencieux, aucun refus de format.
    Explication du passage sans phases 4-5 : les largeurs slicees de l'artefact (5120d,
    vocab 248320, tout multiple de 256) traversent le slicing groupe-256 en dur de
    `slice_uniform_group_cols` — arithmetiquement correct pour qt3 des que `k % 256 == 0`
    (row_bytes = (k/256)*272 ; chaque tranche de 256 colonnes = 8 groupes qt3 alignes),
    verifie par calcul sur les k de l'artefact.
    Consequences : phase 4 se re-calibre en generalisation du groupe par format (qt3 G32
    quelconque, utile pour les artefacts hors G256) + tests de layouts ; phase 5 est
    largement couverte par ce run (decode + prefill court) — restent le prefill long
    (chunks multiples, prompt >= 256 tokens) et la parite tp2-vs-mono (phase 7).
    Debit indicatif : 43.8 tok/s vs baseline mq4 tp2 51.91 tok/s (-15.6 %, un seul run,
    spec off — la receipt reste l'objet de la phase 7).
  - Protocole automatise : `scripts/phase3_preflight_tests.py` pilote la compilation
    (`cargo test -p hipfire-arch-qwen35 --no-run`) puis l'execution des tests (avec
    `--show-output` pour rendre les skips runtime comptables), sous le verrou GPU du plan,
    et produit `phase3-out/run-<ts>/report.md` + `summary.json` + logs bruts ; `--lib-only`
    limite l'execution aux tests unitaires de la lib (aucun GPU), `--filter` passe un filtre
    de nom de test, `--dry-run` affiche les commandes sans les executer.

- **2026-10-27 — Phase 2 validee sur materiel** (implementation `af4cb2f62` ; 2x gfx1100 24 GiB,
  PCIe 28.3 GB/s calibre, hote 123.4 GiB RAM, artefact `qwen3.8-27b.q8.hfq` 26.6 GiB arch 5,
  `memory.max_seq` 32768) :
  - `fit --tp 2` : `topology tp2`, verdict `UNDETERMINED` (capacite par rang decidee au load),
    note `tp2: per-rank weights are not projected (the capacity projection is single-card)`.
  - `fit` mono-carte : `DOES NOT FIT` (poids projetes 30.5 GiB > 23.9 GiB libres) — TP2 est la
    seule route pour cet artefact.
  - `fit --pp 2` : `topology pp2`, KV legacy (refus VMM sous PP, texte inchange) ;
    `fit --tp 2 --pp 2` : refus d'exclusivite pp/tp (texte inchange) ; `--tp 0` / `--tp 65` :
    erreur clap `1..=64`.
  - `run --tp 2` : refus de format attendu au premier tenseur de projection GDN —
    `layers.0.linear_attn.in_proj_qkv.weight: dense TP requires qt13 or qt44, got qt=3`
    (`validate_mq4_proj_info`, `load.rs`). Tout l'amont fonctionne (deux cartes verifiees, locks
    GPU pris, kernels precompiles trouves, KV q8/q8) : le blocage est la porte de format qt3,
    objet des phases 3 a 5.
- **Artefacts de phase 0 identifies** (dans `registry/v1.json`, arch 5) : trunk q8 local
  `qwen3.8-27b.q8.hfq` ; reference mq4 du registre `qwen3.8:27b` -> `qwen3.8-27b.mq4` avec tete
  MTP sidecar `qwen3.8-27b.mtp`. A telecharger et sceller (sha256) a l'execution de la phase 0.
- **2026-10-27 — Phase 0, premier passage sur mq4-xts** : scellement ok (trunk q8, mq4-xts,
  tete MTP ; aucun telechargement) ; `bench mq4 tp2` x3 ok (baseline tp2 mesuree, JSON bruts
  conserves) ; `run mq4 tp2 --spec mtp` ok (attache MTP operationnelle en tp2). Refus de cadrage
  observes (le draft DFlash declare pour les tiers qwen3.8:27b est resolu automatiquement) :
  `run tp2` sans spec -> `EP serving (tp>1) does not support DFlash drafters in v1; reload
  without a draft.` ; `bench pp2` -> `DFlash speculative decode requires pp=1 in v1 (...)` ;
  `HIPFIRE_PP_DFLASH=1` reste ecarte (precedent du plan : charge accepte, generation sans
  spec-decode cross-card). Le script epingle donc `--spec off` (runs/benchs de mesure) et
  `--spec mtp` (run d'attache MTP) ; `--bench-spec` choisit le mode des benchs.
- **2026-10-27 — Phase 0 CLOSE, porte de sortie atteinte** (compte rendu scelle :
  `docs/plans/2026-10-27-qwen35-tp2-phase0-reference.md`, run `20261027T060706Z`) :
  peer access **oui** (`dense qwen TP load complete: 2 ranks, peer_access=true`) ; artefacts
  scelles (trunk q8 26.63 GiB, mq4-xts 13.96 GiB, tete MTP 0.21 GiB, sha256 au compte rendu) ;
  baselines AR (spec off, tg128, 3 processus) : tp2 decode 51.91 tok/s, pp2 decode 44.71 tok/s
  (+16.1 % pour tp2) ; prefill a reprendre avec un prompt >= 256 tokens (24 tokens par defaut =
  overhead de lancement) ; attache MTP tp2 operationnelle (compressed-serial K=3, 68.3 tok/s vs
  51.9 AR sur le run de verification). Les phases 1 et 3 sont debloquees.
- **Protocole automatise** : `scripts/phase0_reference.py` pilote toute la phase (scellement
  sha256, peer access mq4 tp2, attache MTP, 3 processus frais de bench mq4 tp2 et pp2 via
  `HIPFIRE_TOPOLOGY_PP=2`, fit machine) sous le verrou GPU du plan, et produit
  `phase0-out/run-<ts>/report.md` + `summary.json` + logs bruts + JSON bench. `--dry-run`
  affiche les commandes sans les executer. **Aucun telechargement sans `--pull`** : les
  artefacts deja presents dans models/ sont scelles tels quels (`--mq4-file` accepte tout
  fichier mq4* installe, run/bench/fit passent par le chemin) ; sans artefact et sans
  `--pull` le script refuse en listant les candidats. HF_TOKEN est lu par hipfire lui-meme
  (sa presence est consignee, sa valeur jamais affichee).
