;; ============================================================================
;; sde_dvs.cmd — Multi-fin GaN/AlGaN HEMT structure (Sentaurus SDE)
;;
;; Coordinate convention:
;;   x = source -> drain (channel direction)
;;   y = fin axis (fins stacked along +y with spacing Sfin)
;;   z = layer-stack growth direction (top contacts at z = z434)
;;
;; Multi-fin layout:
;;   fin-idx i ∈ [0, Nfins)
;;   y-offset of fin i:  y0_i = i * (Fw + Sfin)
;;   y-top   of fin i:   y5_i = y0_i + Fw
;;   All region names get a "_f<i>" suffix so every fin has distinct regions.
;;   Source/drain AlGaN/GaN spacers use bare P20 names on fin 0 via sd-suffix;
;;   fin>0 gets "_f<i>" suffix (loses 1/8 drain polarization activations).
;;
;; Inter-fin spacer architecture (matches sde_dvs_reference.cmd):
;;   Gold source/drain contacts extend into the inter-fin gap (continuous
;;   Gold bridging adjacent fins). Gate oxide (HfO2) and gate metal (TiN)
;;   also extend into the gap via spacer blocks. AlGaN/GaN semiconductor
;;   stacks remain per-fin — no doped semiconductor in the gap.
;;   This avoids the gate leakage (vth=-14706 V) caused by shared
;;   y-spanning AlGaN spacers contacting gate oxide.
;;
;;   Contacts: S/D have 3 faces per fin (main + 2 Gold spacer tops at z434).
;;   G has 5 faces per fin (3 wraparound + 2 TiN spacer tops at z10).
;;
;; Parameters substituted by src/simulation/simulation.py Simulation._render():
;;   @AlGaN@        AlGaN layer thickness                       [um]
;;   @GaN@          GaN layer thickness (also drives Hbuffer)   [um]
;;   @Tox@          HfO2 gate-oxide thickness                   [um]
;;   @Hg@           TiN gate metal thickness                    [um]
;;   @Fw@           fin width (y extent of a single fin)        [um]
;;   @Lext2@        drain-side spacer length                    [um]
;;   @Nfins@        fin count (integer, >= 1)
;;   @ch_doping@    AlGaN channel doping concentration          [cm^-3]
;;   @AlGaN_doping@ AlGaN source/drain spacer doping            [cm^-3]
;;
;; Hardcoded constants:
;;   Ls = Lext1 = Lg = Ld = 0.1  [um]  source / spacer / gate / drain lengths
;;   Sfin = 0.1                  [um]  inter-fin y-spacing
;;   Al mole fraction = 0.25
;;   Source/Drain Gold contact doping = 1e20 cm^-3
;; ============================================================================


;; ----- Geometric parameters -----
(define Ls    0.1)        ; source contact length
(define Lext1 0.1)        ; source-side spacer length
(define Lg    0.1)        ; gate length
(define Lext2 @Lext2@)    ; drain-side spacer length
(define Ld    0.1)        ; drain contact length

(define Hbuffer  @GaN@)
(define HHbuffer (- @GaN@))
(define HHsub    (- 0.1)) ; substrate slab thickness (negative)

(define Tox  @Tox@)
(define Hg   @Hg@)
(define Fw   @Fw@)
(define Sfin 0.1)         ; inter-fin spacing (fixed — matches SFIN_CONST in simulation.py)
(define Nfins @Nfins@)


;; ----- Derived x-coordinates (shared across all fins) -----
(define x1 Ls)
(define x2 (+ x1 Lext1))
(define x3 (+ x2 Lg))
(define x4 (+ x3 Lext2))
(define x5 (+ x4 Ld))


;; ----- Derived z-coordinates (layer stack, shared across all fins) -----
(define z1   @AlGaN@)
(define z2   (+ z1   @GaN@))
(define z3   (+ z2   @AlGaN@))
(define z33  (+ z3   @GaN@))
(define z333 (+ z33  @AlGaN@))
(define z433 (+ z333 @GaN@))
(define z434 (+ z433 @AlGaN@))   ; top of layer stack (top contact z-face)
(define z4   (+ z434 Tox))       ; oxide top
(define z5   (+ z4   Hg))        ; gate metal top
(define z7   HHbuffer)           ; buffer top (= -GaN, below the stack)
(define z6   (+ z7   Tox))       ; wraparound oxide top
(define z8   (+ z7   HHsub))     ; substrate bottom
(define z10  (+ z6   Hg))        ; gate spacer metal top (inter-fin extension)


;; ----- Doping helper (must be defined before the fin loop) -----
;; Define a constant DopingConcentration profile + 0.25 Al mole fraction
;; profile on each region in the list. Profile names are derived from the
;; region name so they don't collide.
(define (apply-AlGaN-doping regions doping-value)
  (for-each
    (lambda (region)
      (let ((dop (string-append "doping_" region))
            (xm  (string-append "xmole_"  region)))
        (sdedr:define-constant-profile dop "DopingConcentration" doping-value)
        (sdedr:define-constant-profile-region dop dop region)
        (sdedr:define-constant-profile xm  "xMoleFraction" 0.25)
        (sdedr:define-constant-profile-region xm  xm  region)))
    regions))


;; ----- Per-fin geometry + doping -----
(do ((fin-idx 0 (+ fin-idx 1)))
    ((>= fin-idx Nfins))
  (let* ((y0     (* fin-idx (+ Fw Sfin)))
         (y5-f   (+ y0 Fw))
         (y2-f   (- y0 Tox Hg))    ; gate wraparound bottom
         (y3-f   (- y0 Tox))       ; central oxide bottom
         (y6-f   (+ y5-f Tox))     ; central oxide top
         (y7-f   (+ y5-f Tox Hg))  ; gate wraparound top
         (y-ext-lo (- y0 (/ Sfin 2)))  ; spacer lower extent (halfway to prev fin)
         (y-ext-hi (+ y5-f (/ Sfin 2))) ; spacer upper extent (halfway to next fin)
         (suffix (string-append "_f" (number->string fin-idx)))
         ;; Source/drain spacers use bare P20 names on fin 0 to preserve
         ;; Sentaurus's polarization-activating interface ordering.
         ;; Fin>0 gets "_f<i>" suffix → loses 1 of 8 drain activations
         ;; (~11% per-fin current drop). Shared y-spanning source/drain
         ;; was tried but causes gate leakage through doped AlGaN in the
         ;; inter-fin gap.
         (sd-suffix (if (= fin-idx 0) "" suffix)))

    ;; --- Source: Gold contact + 7-layer spacer stack ---
    (sdegeo:create-cuboid (position 0 y0 z7) (position x1 y5-f z434)
                          "Gold" (string-append "SrcC" suffix))
    (for-each
      (lambda (zlo zhi mat base-name)
        (sdegeo:create-cuboid (position x1 y0 zlo) (position x2 y5-f zhi)
                              mat (string-append base-name sd-suffix)))
      (list 0     z1    z2    z3    z33   z333  z433)
      (list z1    z2    z3    z33   z333  z433  z434)
      (list "AlGaN" "GaN"  "AlGaN" "GaN"  "AlGaN" "GaN"  "AlGaN")
      (list "Source132" "Source133" "Source134" "Source135" "Source136" "Source137" "Source138"))

    ;; --- Gate: TiN metal + HfO2 oxide (wraparound + central; two-cuboid merge) ---
    (sdegeo:create-cuboid (position x2 y2-f z7) (position x3 y7-f z5)
                          "TiN" (string-append "Gate" suffix))
    (sdegeo:create-cuboid (position x2 y3-f z6) (position x3 y6-f z4)
                          "HfO2" (string-append "GateOx" suffix))
    (sdegeo:create-cuboid (position x2 y2-f z7) (position x3 y7-f z6)
                          "HfO2" (string-append "GateOx" suffix))

    ;; --- Channel: 7-layer AlGaN/GaN stack under the gate ---
    (for-each
      (lambda (zlo zhi mat base-name)
        (sdegeo:create-cuboid (position x2 y0 zlo) (position x3 y5-f zhi)
                              mat (string-append base-name suffix)))
      (list 0     z1    z2    z3    z33   z333  z433)
      (list z1    z2    z3    z33   z333  z433  z434)
      (list "AlGaN" "GaN"  "AlGaN" "GaN"  "AlGaN" "GaN"  "AlGaN")
      (list "ChA1"  "ChG1" "ChA2"  "ChG2" "ChA3"  "ChG3" "ChA4"))

    ;; --- Drain: 7-layer spacer stack + Gold contact ---
    (for-each
      (lambda (zlo zhi mat base-name)
        (sdegeo:create-cuboid (position x3 y0 zlo) (position x4 y5-f zhi)
                              mat (string-append base-name sd-suffix)))
      (list 0     z1    z2    z3    z33   z333  z433)
      (list z1    z2    z3    z33   z333  z433  z434)
      (list "AlGaN" "GaN"  "AlGaN" "GaN"  "AlGaN" "GaN"  "AlGaN")
      (list "Drain131" "Drain132" "Drain133" "Drain134" "Drain135" "Drain136" "Drain137"))
    (sdegeo:create-cuboid (position x4 y0 z7) (position x5 y5-f z434)
                          "Gold" (string-append "DrC" suffix))

    ;; --- Sub-channel GaN buffer slabs under each x-region ---
    (for-each
      (lambda (xlo xhi base-name)
        (sdegeo:create-cuboid (position xlo y0 0) (position xhi y5-f z7)
                              "GaN" (string-append base-name suffix)))
      (list x1 x2 x3)
      (list x2 x3 x4)
      (list "Bulk1" "Bulk2" "Bulk3"))

    ;; --- Per-fin doping ---
    (apply-AlGaN-doping
      (list (string-append "ChA1" suffix) (string-append "ChA2" suffix)
            (string-append "ChA3" suffix) (string-append "ChA4" suffix))
      @ch_doping@)
    (apply-AlGaN-doping
      (list (string-append "Source132" sd-suffix) (string-append "Source134" sd-suffix)
            (string-append "Source136" sd-suffix) (string-append "Source138" sd-suffix))
      @AlGaN_doping@)
    (apply-AlGaN-doping
      (list (string-append "Drain131" sd-suffix) (string-append "Drain133" sd-suffix)
            (string-append "Drain135" sd-suffix) (string-append "Drain137" sd-suffix))
      @AlGaN_doping@)
    (let* ((src-name (string-append "SrcC" suffix))
           (dop-name (string-append "doping_" src-name)))
      (sdedr:define-constant-profile dop-name "ArsenicActiveConcentration" 1e20)
      (sdedr:define-constant-profile-region dop-name dop-name src-name))
    (let* ((dr-name  (string-append "DrC" suffix))
           (dop-name (string-append "doping_" dr-name)))
      (sdedr:define-constant-profile dop-name "ArsenicActiveConcentration" 1e20)
      (sdedr:define-constant-profile-region dop-name dop-name dr-name))

    ;; --- Inter-fin spacer blocks ---
    ;; Gold source/drain spacers connect fins; gate oxide/metal extends
    ;; the wraparound into the gap.  AlGaN/GaN stacks stay per-fin (no
    ;; semiconductor in the gap → no gate leakage).

    ;; Gate oxide spacers (HfO2, extend wraparound into inter-fin gap)
    (sdegeo:create-cuboid (position x2 y-ext-lo z7) (position x3 y0 z6)
                          "HfO2" (string-append "GateOxB" suffix))
    (sdegeo:create-cuboid (position x2 y5-f z7) (position x3 y-ext-hi z6)
                          "HfO2" (string-append "GateOxA" suffix))

    ;; Gate metal spacers (TiN, z6..z10 slab only)
    (sdegeo:create-cuboid (position x2 y-ext-lo z6) (position x3 y2-f z10)
                          "TiN" (string-append "GateExtB" suffix))
    (sdegeo:create-cuboid (position x2 y6-f z6) (position x3 y-ext-hi z10)
                          "TiN" (string-append "GateExtA" suffix))

    ;; Gold source spacers (connect source Gold across fins)
    (sdegeo:create-cuboid (position 0 y-ext-lo z7) (position x1 y0 z434)
                          "Gold" (string-append "SrcCB" suffix))
    (sdegeo:create-cuboid (position 0 y5-f z7) (position x1 y-ext-hi z434)
                          "Gold" (string-append "SrcCA" suffix))

    ;; Gold drain spacers (connect drain Gold across fins)
    (sdegeo:create-cuboid (position x4 y-ext-lo z7) (position x5 y0 z434)
                          "Gold" (string-append "DrCB" suffix))
    (sdegeo:create-cuboid (position x4 y5-f z7) (position x5 y-ext-hi z434)
                          "Gold" (string-append "DrCA" suffix))

    ;; Gold spacer doping
    (for-each
      (lambda (region-name)
        (let ((dop-name (string-append "doping_" region-name)))
          (sdedr:define-constant-profile dop-name "ArsenicActiveConcentration" 1e20)
          (sdedr:define-constant-profile-region dop-name dop-name region-name)))
      (list (string-append "SrcCB" suffix) (string-append "SrcCA" suffix)
            (string-append "DrCB" suffix)  (string-append "DrCA" suffix)))))


;; ----- Wide GaN buffer slab spanning the full device width below the layer stack -----
;; Extends in y from below fin 0's gate wraparound to above fin (Nfins-1)'s.
(let* ((y-bot (- (- 0 Tox) Hg 0.1))
       (y-top (+ (* (- Nfins 1) (+ Fw Sfin)) Fw Tox Hg 0.1)))
  (sdegeo:create-cuboid (position 0 y-bot z7) (position x5 y-top z8)
                        "GaN" "Buffer"))


;; ----- Contacts -----
;; S, D: 3 faces per fin (main Gold top + 2 Gold spacer tops at z434).
;; G: 5 faces per fin (3 wraparound + 2 TiN spacer tops at z10).

(sdegeo:define-contact-set "S" 4.0 (color:rgb 1.0 0.0 0.0) "##")
(sdegeo:set-current-contact-set "S")
(do ((fin-idx 0 (+ fin-idx 1)))
    ((>= fin-idx Nfins))
  (let* ((y0   (* fin-idx (+ Fw Sfin)))
         (y2-f (- y0 Tox Hg))
         (y7-f (+ y0 Fw Tox Hg)))
    (sdegeo:set-contact-faces (find-face-id (position 0.001 (+ y0 0.001) z434)))
    (sdegeo:set-contact-faces (find-face-id (position 0.001 y7-f z434)))
    (sdegeo:set-contact-faces (find-face-id (position 0.001 y2-f z434)))))

(sdegeo:define-contact-set "D" 4.0 (color:rgb 1.0 0.0 0.0) "##")
(sdegeo:set-current-contact-set "D")
(do ((fin-idx 0 (+ fin-idx 1)))
    ((>= fin-idx Nfins))
  (let* ((y0   (* fin-idx (+ Fw Sfin)))
         (y2-f (- y0 Tox Hg))
         (y7-f (+ y0 Fw Tox Hg)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x4 0.001) (+ y0 0.001) z434)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x4 0.001) y7-f z434)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x4 0.001) y2-f z434)))))

;; Gate: 5 faces per fin (front/top/back wraparound + 2 ext spacer tops at z10)
(sdegeo:define-contact-set "G" 4.0 (color:rgb 1.0 0.0 0.0) "||")
(sdegeo:set-current-contact-set "G")
(do ((fin-idx 0 (+ fin-idx 1)))
    ((>= fin-idx Nfins))
  (let* ((y0   (* fin-idx (+ Fw Sfin)))
         (y2-f (- y0 Tox Hg))
         (y7-f (+ y0 Fw Tox Hg)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x2 0.001) y2-f             z4)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x2 0.001) (+ y0 0.001)     z5)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x2 0.001) y7-f             z4)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x2 0.001) y7-f             z10)))
    (sdegeo:set-contact-faces (find-face-id (position (+ x2 0.001) y2-f             z10)))))


;; ----- Mesh refinement -----
;; Global refinement size (shared by all fins)
(sdedr:define-refinement-size "Cha_Mesh" 0.008 0.008 0.008 0.02 0.02 0.02)
(sdedr:define-multibox-size    "multiboxSizeChannel" 0.2 0.2 0.01 0.01 0.01 0.01)

;; Per-fin mesh refinement window + multibox placement
(do ((fin-idx 0 (+ fin-idx 1)))
    ((>= fin-idx Nfins))
  (let* ((y0     (* fin-idx (+ Fw Sfin)))
         (y5-f   (+ y0 Fw))
         (idx    (number->string fin-idx))
         (win    (string-append "multiboxChannel_f"   idx))
         (plc    (string-append "multiboxPlacement_f" idx)))
    (sdedr:define-refinement-window win "Cuboid"
      (position x1 y0 z7) (position x4 y5-f z434))
    (sdedr:define-multibox-placement plc "multiboxSizeChannel" win)
    (sdedr:define-refinement-function plc "DopingConcentration" "MaxTransDiff" 1)))

;; Global material-interface refinement
(sdedr:define-refinement-function "RS.global" "MaxLenInt" "GaN" "AlGaN" 0.001 "DoubleSide")


;; ----- Build mesh -----
(sde:build-mesh "" "node")
