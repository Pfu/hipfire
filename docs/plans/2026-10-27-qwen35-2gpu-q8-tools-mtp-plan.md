# Qwen3.5 (arch 5) — 2 GPU + tool_call + MTP en Q8 : plan d'implementation ordonne

**Status:** actif — phases 0, 2, 3, 4, 5 et 6 closes sur materiel (journal et compte rendu plus bas ; phase 4 : slicing par groupe de format ; phase 5 : admission kernels Q8 en decode et prefill long >= 256 tokens ; phase 6 : capture whole-token Q8 rendue observable et attache MTP tp2 — 49.7 tok/s indicatif —, voie AR/auto intacte) ; phase 1 close (superseedee par les preuves des phases 3-6) ; phase 7 livree cote depot (criteres d'acceptation fixes au journal, protocole de session, docs) — validation GPU en attente (oracle mesh, serve outils, non-regression mq4, receipt de debit) ; phases 8-9 non demarrees.
**Auteur:** (agent)
**Cree:** 2026-10-27
**Specification:** `docs/plans/2026-10-27-qwen35-2gpu-q8-tools-mtp-spec.md` — absente du
depot et non retrouvee (verifie le 2026-10-27, decision avec l'utilisateur) : les criteres
d'acceptation de la phase 7 sont fixes explicitement a l'entree « Phase 7 LIVREE » du journal
ci-dessous et au protocole de session.

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

- **2026-10-27 — Phase 7 LIVREE cote depot (criteres fixes, protocole de session, docs ; validation GPU en attente)** :
  - Decisions prises avec l'utilisateur (la spec citee par la porte de sortie etant absente du
    depot et non retrouvee) :
    - Point 1 (parite mesh-vs-mono) — option (a) : l'oracle existant
      (`qwen35_tp2_vs_single_oracle`, `qwen35_pp2_vs_single_oracle`) tourne sur la reference
      mq4-xts scellee (deja la fixture epingulee du harnais, intacte dans models_dir) ; la
      parite qt3-mono sur le trunk Q8 27B n'est pas mesurable (fit mono `DOES NOT FIT`,
      phase 2) et est documentee comme telle. Justification : le harnais epingle taille+sha256
      en dur (`HIPFIRE_QWEN35_FIXTURE` ne surcharge que le chemin), aucun artefact qwen35 qt3
      n'existe au registre (94 entrees arch 5 : mq3/mq4/mq5/mq6), et le chargement qt3 en
      route mono n'a jamais ete prouve — l'option (b) (petit artefact qt3) aurait exige un
      epinglage surchargeable (code + test), un run de quantization et la preuve mono-qt3,
      hors perimetre phase 7. Le risque numerique residual cote Q8 reste couvert par les
      phases 3-6 (admission sans refus, determinisme greedy mot a mot, capture whole-token
      sans etat partiel). L'option (b) reste rejouable independamment de la phase 7.
    - Point 6 (spec section 5) — criteres d'acceptation fixes explicitement (liste ci-dessous) ;
      en-tete du plan corrige pour ne plus promettre un fichier fantome.
    - Phase 1 (spike) — close comme superseedee (entree ci-dessous).
  - **Criteres d'acceptation de la phase 7 (liste fermee, fixes au 2026-10-27)** :
    1. Oracle mesh-vs-mono : `qwen35_tp2_vs_single_oracle` vert, 3 processus frais, sur la
       reference mq4-xts scellee (critere « tokens `assert_eq` + |Δlogit| <= 1.0 »,
       `BOUND_TP_ABS`), sous flock ; `qwen35_pp2_vs_single_oracle` : 1 processus, complement ;
       non-mesurabilite qt3-mono sur 27B documentee (fit `DOES NOT FIT`, phase 2).
    2. Serveur E2E trunk Q8 : `hipfire serve <trunk q8> --tp 2 --spec mtp --max-seq 4096` ;
       requete `tools` => `finish_reason: "tool_calls"` ; requete `stop` => arret honore sans
       fuite du texte ; requete hors budget => refus nommant la limite (`request exceeds
       advertised context window: ...` ou `request exceeds loaded KV budget: ...`, verbatim).
    3. Non-regression mq4 tp2 : meme harnais sur mq4-xts (serve + tools + stop) + controle AR
       de decode dans la signature phase 0 (~52 tok/s).
    4. Docs : `docs/multi-gpu.md` (formats admis TP dense), `docs/speculation-support-inventory.md`
       (MTP sur mesh dense), `CHANGELOG.md` ; `docs/CONFIG.md` sans modification (aucun message
       de refus cite).
    5. Receipt de debit horodate : decode AR + MTP (tg128, 3 processus frais), prefill
       prompt-a/b/c (`--max-tokens 1`, comptes de tokens reels — ancre trace ~558/1850/3515),
       avec binaire (version + commit) et identite du modele (chemin + sha256 scelle).
  - Livre au depot : `docs/multi-gpu.md` (section « Dense Qwen3.5/3.6 TP: admitted weight
    formats »), `docs/speculation-support-inventory.md` (bullet mesh dense etendu : formats
    admis, refus n-gram nomme, validation materielle phase 6), `CHANGELOG.md` (entree
    Unreleased), en-tete et journal du present plan. `docs/CONFIG.md` : non modifie (aucun
    message de refus cite).
  - Protocole GPU : `results/session-20261027-phase7/protocole-e2e-tools-receipt.md` (dossier
    local, non versionne) — etape 0 CPU, oracle mesh (3 processus tp2 + 1 pp2), serve E2E
    (tools / stop / check max_seq, prompts non vides, chemins absolus), non-regression mq4,
    receipt de debit ; commandes GPU sous `flock -w 3600 /tmp/hipfire-gpu.lock`.
  - Porte de sortie : les 5 criteres verts et ce journal a jour => phase 7 close ; la cible
    « 2 GPU + outils + MTP en Q8 » est alors declaree atteinte. Phases 8 (outils sur PP) et 9
    (MTP sur mesh PP) non demarrees, conformement au plan.

- **2026-10-27 — Phase 1 CLOSE (superseedee)** : l'objet de la phase 1 (liste exhaustive
  `(op, M, K, admis/refuse, message)` pour les formes locales Q8 du forward dense-TP) a ete
  prouve sur materiel par les phases 3-6 : tous les kernels Q8 admis en decode et en prefill
  long (trois regimes de chunk, phase 5), capture whole-token complete sans etat partiel
  (phase 6), ensemble de kernels du speculateur MTP admis au vol sans refus. Aucun refus
  structurel a traiter : la branche « formes refusees => la phase 5 inclut leur admission »
  n'a jamais ete empruntee. Aucun code ni artefact de spike a rapatrier (la phase 1 n'a rien
  livre, conformement a son « Interdit ») ; le contenu decisionnel est porte par le journal
  des phases 3-6. Close sans changement de code.

- **2026-10-27 — Phase 6 CLOSE : capture whole-token Q8 rendue observable + attache MTP tp2, voie AR/auto intacte** :
  - Capture : `HIPFIRE_AR_GRAPH_TRACE=1` rend observable la capture decode-time du graphe
    whole-token (deja engagee par defaut sur la paire exacte gfx1100 tp2, `HIPFIRE_TP_FULL_GRAPH`
    unset, mais muette depuis la phase 3) — une ligne par processus, au premier token propre :
    `[qwen-dense-tp-graph] captured whole-token tp=2 barriers=128 pos=560` (prompt-a) puis
    `pos=12` ("Hello") ; aucun `TP2 whole-token graph state is partial`, aucun fallback, aucun
    refus, sorties coherentes (43.5 et 44.0 tok/s). Contre-epreuve `HIPFIRE_TP_FULL_GRAPH=0`
    (route segmentee) non jouee — optionnelle au protocole.
  - Attache MTP : apres restoration de la tete scellee — supprimee du disque pendant la
    session (un lien `.q8.mtp` devenu casse a produit un refus fail-closed colate :
    `MTP head required (mtp=on) but not loaded: no bundled trailer or .mtp sidecar found`,
    rollback propre avant upload des poids) — `hipfire pull qwen3.8:27b-mq4-xts --yes --ple
    none` a rapatrie `qwen3.8-27b.mtp` au sha256 exact du scelle phase 0 (les 13 tiers du
    registre declarant la meme tete ; le trunk mq4-xts present n'a pas ete re-telecharge) et
    le lien `<trunk>.mtp` redevient valide. Les trois logs d'attache attendus sont presents
    verbatim (`MTP head loaded (sidecar ..., dense TP rank 0): n_embd=5120 vocab=248320`,
    `qwen35 MTP speculator enabled (compressed-serial, K=3)`, `qwen35 MTP prompt fill route:
    ar (AR prefill route)`) et le decode speculatif tourne : `[req run] drafter=mtp tau=2.00
    tok/s=49.7 decode (108 tok, 36 windows)` — 3.0 tok/fenetre (1 trunk + 2.0 drafts acceptes
    de 3), +14-15 % vs AR 43.2-43.6 (indicatif, prompt vide ; receipt phase 7 ; phase 0 sur
    mq4-xts : +31.6 %). Nouvel ensemble de kernels du speculateur admis au vol sans refus
    (tete mq4-G256 : gemm_hfq4g256, gemv_hfq4g256_rdna3, mq_rotate_x, fused_*_mq_rotate,
    argmax_token_chain, argmax_f32_batched, greedy_accept, dflash_gdn_pre,
    dflash_state_bulk_copy, gemm_q8_0_wmma, kv_cache_write_q8_0_batched).
  - Voie AR/auto intacte : defaut (auto) — AUCUNE ligne MTP ; observable nouveau, explique par
    le code : `  dense TP: n-gram drafter refused (block verify is not wired on the mesh)`
    (`crates/hipfire-loader/src/lib.rs:5405-5414` — le site de chargement dense-TP force
    `ngram_draft: Some(false)` et refuse nommement le bras n-gram ; DFlash est passe None en
    dur sur le mesh, d'ou l'absence du refus DFlash v1 attendu d'apres la phase 0 : sur le
    mesh dense-TP, le defaut ne peut qu'etre AR). Controle AR `--spec off` "Hello" : 54 tok @
    44.0 tok/s — la signature des clotures 3/4. Determinisme greedy : la reponse a prompt-a
    est identique mot a mot en AR trace, en auto (n-gram refuse) et au run phase 5.
  - Collation comptes de tokens (correction du protocole phase 5) : la trace de capture ancre
    prompt-a a ~557-559 tokens (462 mots ≈ 1.21 tok/mot) — les estimations phase 5
    (~350/1150/2200, ratio inverse) etaient sous-comptees d'un facteur ~1.6 : regimes corriges
    ≈558 (chunk partiel), ≈1850 (1 plein + tail), ≈3515 (3 pleins + tail, et non 2) pour les
    runs phase 5 ; l'admission par hauteur de chunk reste acquise.
  - Contexte artefacts : trunk et tete vivent desormais dans models_dir (`qwen3.8-27b.q8.hfq`,
    lien `qwen3.8-27b.q8.mtp` -> tete scellee) ; la reference `qwen3.8-27b.mq4-xts` est
    intacte (non-regression mq4 de la phase 7). Aucun changement de code, aucun test ajoute
    (memes raisons que la phase 5) ; chemins MQ4 intouches (filtres verts phase 5).
  - Transcriptions : `results/session-20261027-phase6/` (dossier local, non versionne). Porte
    de sortie atteinte (« un decode MTP fonctionne sur trunk Q8 en tp2, avec le log d'attache
    attendu ; la voie AR reste intacte ») → phase 7 (parite tp2-vs-mono, serveur outils, docs,
    receipt) debloquee.

- **2026-10-27 — Phase 5 CLOSE : admission kernels Q8 en decode et prefill long sur materiel** :
  trois runs `run /home/spf/qwen3.8-27b.q8.hfq --tp 2 --max-seq 4096 --temp 0 --spec off` sous le
  verrou du plan, daemon installe a jour (plus de `HIPFIRE_DAEMON_BIN`), artefact scelle en
  phase 0 — en lieu et place du run unique + 3 benchs du protocole : trois longueurs de prompt
  croissantes couvrent les memes regimes de chunk (cap tp2 = 1024 tokens, `prefill_max_batch_tp`
  = 512*2) avec un decode reel en plus. Aucun refus, aucune ligne `unsupported`, aucun fallback
  silencieux ; `hipfire load_phases.v1` nominale a chaque run (~26.9 s, 50.97 GB lus, 0 major
  fault) ; meme ensemble de paquets de kernels indexes verifies qu'aux clotures de phases 3/4,
  en decode ET en prefill (embedding_q8_batched, rmsnorm, gemm_qkvza_q8_0_wmma,
  gated_delta_net_q8_fast, gemm_q8_0_residual_wmma, gemm_gate_up_q8_0_wmma, gemm_qkv_q8_0_wmma,
  attention_q8_0_flash_prefill_wmma_gfx11_hd256, attention_flash_q8_0_tile/reduce, gemv_q8_0,
  conv/rope/silu/gated_norm et helpers).
  - run a (prompt-a.txt, 462 mots, ~350 tokens estimes — chunk partiel attendu) : 1870 tok @
    43.6 tok/s ; collation exacte de la structure du prompt par le modele (7 paragraphes de 4
    phrases, le dernier a 3 — verifie dans le fichier) et des details cites (trip count, lime
    content, shutters, offset in his log — tous presents ; deux fragments cites paraphrases,
    absents des trois prompts, sans portee).
  - run b (prompt-b.txt, 1533 mots, ~1100-1180 estimes — 1 chunk plein + tail attendus) :
    852 tok @ 43.4 tok/s ; le modele decrit la boucle du texte et elle est exacte (l'ouverture
    « The harbor lights flickered twice » apparait 2 fois, fin sur les rain barrels — verifie
    dans le fichier).
  - run c (prompt-c.txt, 2916 mots, ~2100-2200 estimes — 2 chunks pleins + tail attendus) :
    extension du mapping KV pendant le prefill (`mapped_prefix=3855->4096 / physical_cap=4096`),
    496 tok @ 43.2 tok/s, reponse coherente sur le contenu du prompt long (collation partielle :
    transcription tronquee ; decompte « roughly sixty » du modele non colle au decompte
    mecanique — sans portee sur l'admission).
  - Comptes de tokens estimes (pas de tokenizer cote agent) ; les logs des runs n'impriment pas
    les bornes de chunk : l'observable dur est trois prefills longs (>= 256 tokens estimes)
    sans aucun refus, dont deux superieurs au cap de chunk estime (1024).
  - Complements CPU : audit `scripts/phase4_slice_arithmetic.py --hfq` — 498 tenseurs denses
    verifies, 0 hors G256, 0 echec ; `cargo test -p hipfire-arch-qwen35` et filtres `qt13` /
    `qt44` / `dense_tp` verts — chemins MQ4 intouches (point 3 du travail demande).
  - Aucun test cible par op ajoute : aucun code de production n'a change en phase 5 (l'admission
    est un fait d'execution GPU observable dans les logs, pas une surface de code nouvelle) ;
    seams GPU existantes inchangees.
  - Benchs `--prompt-file --max-tokens 1` non executes : la mesure de debit prefill
    (`prefill_tok_s`, comptes de tokens reels) reste l'objet de la phase 7 (receipt).
  - Debits indicatifs AR (spec off) : 43.2-43.6 tok/s, en ligne avec les clotures 3/4
    (43.8-44.0) ; baseline mq4 tp2 51.91 tok/s.
  - Transcriptions : `results/session-20261027-phase5/gpu-run-{a,b,c}.md` (dossier local, non
    versionne). Porte de sortie atteinte (« un run decode et un run prefill complets sur un
    trunk q8 en --tp 2, sans erreur d'admission, sur formes reelles ») → phase 6 (graphe TP2 +
    attache MTP sur trunk Q8) debloquee.

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
