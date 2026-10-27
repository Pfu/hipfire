# Qwen3.5 (arch 5) — 2 GPU + tool_call + MTP en Q8 : plan d'implementation ordonne

**Status:** draft — a valider avant la phase 1.
**Auteur:** (agent)
**Cree:** 2026-10-27
**Specification:** `docs/plans/2026-10-27-qwen35-2gpu-q8-tools-mtp-spec.md`

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
