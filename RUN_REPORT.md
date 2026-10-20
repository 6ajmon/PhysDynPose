# PhysDynPose — próba uruchomienia i ocena przydatności badawczej

Data: 2026-10-20
Środowisko: Windows 11 + WSL2 (Ubuntu), CPU Ryzen (8 wątków), GPU **AMD Radeon RX 6750 XT** (brak CUDA), 16 GB RAM.
Repozytorium: `C:\Github\PhysDynPose` (commit `97569d8`).

---

## 1. Podsumowanie w jednym akapicie

Pipeline **da się uruchomić** — ale nie w całości na tym sprzęcie. Etapy 4–6 (`align_slam.py` → `run_opt.py` → `convert_res.py`) przeszły end-to-end na danych syntetycznych w środowisku WSL2/CPU i są **bardzo tanie obliczeniowo** (~7 ms/frame, ok. 10 s na 1000 klatek). Etapy 1–3 (`extract_movicam_frames.py`, 4D-Humans, DROID-SLAM) są zablokowane: DROID-SLAM **wymaga CUDA** (≥11 GB), 4D-Humans/HMR2 na CPU jest praktycznie nieużywalny, a pełny dataset jest za protekcją anty-botową Edmonda i wymaga licencyjnych plików SMPL. Kluczowy wniosek dla pracy badawczej: **jeśli macie ground-truth pozę kamery (a MoviCam ma), SLAM jest zbędny** — jest do tego gotowy przełącznik `--no-slam`, a ścieżka SLAM w wydanym kodzie i tak jest alignowana do GT i zawiera „magiczny” współczynnik skali `0.6`, który na idealnej trajektorii psuje wynik o ~1.2 m.

---

## 2. Co udało się uruchomić, a co nie

| # | Etap | Status | Przyczyna / uwagi |
|---|------|--------|-------------------|
| 1 | `extract_movicam_frames.py` (krojenie wideo) | ⚪ nie uruchamiany | brak wideo z datasetu (Edmond za Anubis, duży archiwum) |
| 2 | 4D-Humans (`demo.py` → `out_list.pkl`) | 🔴 zablokowany | wymaga CUDA w praktyce (ViT-H + ViTDet + detectron2/mmcv); na CPU rzędu s/frame; dodatkowo licencyjny model SMPL |
| 2b | **FrankMocap** (body) → `out_list.pkl` | 🟢 działa (CPU) | adapter [`tools/frankmocap_to_out_list.py`](C:/Github/PhysDynPose/tools/frankmocap_to_out_list.py), **6–7 fps na CPU**; szczegóły w §13 |
| 3 | DROID-SLAM (`run_droid_slam.py`) | 🔴 zablokowany | oficjalny wymóg: **GPU z CUDA ≥ 11 GB**; brak ROCm dla RX 6750 XT |
| 4 | `align_slam.py --no-slam` | 🟢 działa | 1000 klatek: **2.5 s** |
| 5 | `run_opt.py --no-slam` | 🟢 działa | 1000 klatek: **7.3 s** (~7 ms/klatkę, 358 MB RSS) |
| 6 | `convert_res.py --no-slam` | 🟢 działa | 1000 klatek: **0.66 s** (głównie import smplx) |
| 7 | `visualizer.py` | ⚪ nie testowany | tylko tryb `p.GUI` (PyBullet), potrzebny pulpit; brak trybu headless |

Pomiary dotyczą **danych syntetycznych** (patrz §3) — mierzą koszt kodu, nie jakość wyników.

---

## 3. Jak powstał działający test (i czego w nim brakuje)

Nie da się legalnie pobrać modeli SMPL bez rejestracji na smpl.is.tue.mpg.de, a datasetu nie pobieraliśmy. Dlatego:

1. Wygenerowano **syntetyczny model w formacie SMPL** (24 stawy, 6890 wierzchołków, poprawna hierarchia, kanoniczna rama: miednica w początku układu) — spełnia wszystkie wymagania `smplx` i `articulate.ParametricModel`, ale geometria jest losowa.
2. Wygenerowano pliki wejściowe w **formacie MoviCam**: `camera_exts_full.txt` (bloki 3-liniowe), `cam_ext.npy` (N,4,4), `pred_con_gt.npy` (N,4), `inputs/4d_humans/outputs_XX/out_list.pkl` w formacie 4D-Humans, `pip_reference/.../shape.pt`.
3. Puszczono przez to `align_slam.py`, `run_opt.py`, `convert_res.py`.

Wynik: wszystkie trzy etapy kończą się sukcesem, tensory są skończone, QP się rozwiązuje. **Liczby nie mają sensu fizycznego** — to test ścieżek kodu i wydajności, nie metody.

Dwie pułapki wykryte po drodze (ważne przy własnych danych):

- `align_slam.py` kroi intrinsicsy do zakresu klatek z `config.FRAME_RANGES`, ale **`cam_ext.npy` musi być już przycięty** do tego samego zakresu. Jeśli wrzucicie pełny plik, `min(len(intrinsics), len(cam_ext), len(tracks))` cicho obetnie złe klatki.
- `convert_res.py` ładuje `data/body_models/smpl/SMPL_NEUTRAL.pkl` (domyślny gender) **oraz** `data/J_regressor_extra.npy`, mimo że woła `orig_joints=True` (czyli tego regressora nie używa). Brak pliku = `AssertionError`.

---

## 4. Środowisko: co nie działało i jak to naprawiłem

Repo zakłada Linuksa + CUDA. Na Windowsie z AMD GPU trzeba iść przez WSL2.

| Problem | Objaw | Rozwiązanie |
|---------|-------|-------------|
| Python 3.13 na Windows | `torch==2.1.0` i `pytorch3d` nie mają wheeli dla cp313; pytorch3d nie ma wheeli Windows w ogóle | WSL2 Ubuntu + Miniconda + Python 3.10 |
| brak `pip`/`venv` w WSL | `ensurepip is not available` | Miniconda (instalacja do `$HOME`, bez sudo) |
| nowa conda wymaga akceptacji ToS kanałów Anacondy | `CondaToSNonInteractiveError` | `-c conda-forge --override-channels` |
| `pybullet==3.2.5` | brak wheela, kompilacja wymaga `g++` | `conda install -c conda-forge pybullet` |
| `cvxopt==1.2.6` | kompilacja wymaga BLAS/LAPACK | `conda install -c conda-forge cvxopt` (1.3.3) |
| RBDL (bindings Pythona) | nie ma na PyPI; pakiet conda-forge `rbdl` to tylko C++ | build ze źródeł: `git clone --recursive -b v3.3.1`, `-DRBDL_BUILD_PYTHON_WRAPPER=ON`, Cython + numpy headers; `rbdl.so` skopiowany do site-packages, `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| `pytorch3d==0.7.5` | **nie istnieje na PyPI** (`No matching distribution`) | w smoke-teście zastąpiony 20-liniowym shimem dwóch funkcji; docelowo build ze źródeł na maszynie z CUDA |
| `chumpy==0.71` | **nie istnieje na PyPI** (max 0.70), a 0.70 nie importuje się z numpy ≥ 1.24 (`from numpy import bool`) | potrzebny patch/ fork chumpy albo pickle SMPL bez chumpy; w smoke-teście nie było potrzebne |

Sprawdzone, działające wersje: Python 3.10.22, torch 2.1.0+cpu, numpy 1.26.4, scipy 1.10.1, smplx 0.1.28, pybullet 3.2.5, qpsolvers 4.13 + clarabel 0.11.1, cvxopt 1.3.3, RBDL 3.3.1 (zbudowany lokalnie).

### Ważne: solver QP

`dynamics_scene.py` najpierw woła **Clarabel**, a przy porażce **cvxopt**. Na części klatek Clarabel zwraca `None` (np. gdy stopy są poniżej podłogi) i bez cvxopt kod kończy się `KeyError: 'cvxopt'`. To nie jest opcjonalna zależność „nice to have” — **cvxopt jest wymagany**.

---

## 5. Czy można zignorować SLAM? — TAK

### Co mówi kod

- `run.sh --no-slam` → `align_slam.py --no-slam` → `camera_poses = np.load(cam_ext.npy)` (ground-truth extrinsic kamery z Captury), zamiast `load_slam_poses()`.
- Co ważniejsze: nawet ścieżka `--slam` **nie jest samodzielna**. `align_orb_poses()` skaluje trajektorię SLAM względem GT (`gt_scale / orb_scale`) i zakotwicza ją w pierwszej pozycji GT. SLAM służy więc do „wygładzenia/uzupełnienia” toru kamery, a nie do wyznaczenia skali.

### Dowód eksperymentalny

Podaliśmy `align_slam.py --slam` trajektorię **identyczną z GT** (pozornie idealny SLAM). Wynik:

```
GT displacement        1.998 m
"SLAM" displacement    1.199 m      ← dokładnie ×0.6
rotation diff          0.0°
różnica translacji korzenia po optymalizacji: mean 1.17 m, max 2.32 m
```

Źródło: `align_slam.py`, `align_orb_poses()`:

```python
scale = gt_scale / orb_scale * 0.6          # magiczny współczynnik
rotation_fix = np.array([[0,0,1],[0,-1,0],[1,0,0]])   # zamiana osi
```

Wniosek: **przy własnych danych z GT kamerą zawsze używajcie `--no-slam`**. Ścieżka `--slam` ma stałe dopasowane do konkretnej konwencji (ORB-SLAM3 / ich danych) i wprowadza błąd skali 40% oraz obrót osi; na nowych danych trzeba ją re-kalibrować.

---

## 6. Czy model „skwantyzowany” jest użyteczny?

W repo **nie ma żadnego skwantyzowanego checkpointu** i — co ważne — **PhysDynPose nie ma własnych wag**. `net.py` to opakowanie na `PhysicsOptimizer`, czyli solver QP (Clarabel/cvxopt). Nie ma czego kwantyzować. „Model” może więc oznaczać dwie rzeczy:

### 6a. 4D-Humans / HMR2 (jedyny sieciowy model w pipeline)

- To ViT-H (~630 M parametrów) + ViTDet, uruchamiany w fp32.
- **fp16**: w praktyce dla ViT zwykle < 1% różnicy w parametrach SMPL — bezpieczne, jeśli sprzęt wspiera.
- **int8 (PTQ)**: realne ryzyko przesunięcia rotacji o kilka stopni i błędu głębokości; nie zmierzyłem tego — na tym sprzęcie nie ma CUDA, a instalacja 4D-Humans (detectron2/mmcv/pytorch3d) na CPU jest niepraktyczna.
- Co łagodzi błędy: `align_slam.py` wygładza translację `gaussian_filter1d(sigma=6)` i rotacje filtrem One-Euro, a `run_opt.py` rzutuje ruch na fizycznie dopuszczalny. Szum HMR2 jest więc częściowo absorbowany, ale **błąd systematyczny (skala głębokości) przechodzi do wyniku**, bo `pred_cam_t_full` definiuje translację świata.
- Co *nie* zależy od kwantyzacji HMR2: ostateczny kształt ciała w `convert_res.py` pochodzi z `pip_reference/.../shape.pt`, nie z `betas` HMR2.

**Proponowany eksperyment (do wykonania na maszynie z CUDA):**
1. fp32 vs fp16 vs int8 (TensorRT/ONNX Runtime) na tych samych klatkach.
2. Metryki: Δ(MVPJE) po `convert_res.py` względem GT joints; Δ root translation; Δ global orientation.
3. Raportować osobno błąd detekcji (ViTDet) i błędu regresji HMR2 — int8 na detektorze może gubić klatki.

### 6b. `models/SMPL_male_10PCs.pkl` (10 współczynników PCA)

To nie kwantyzacja, tylko **kompresja PCA**, ale skutki są policzalne:

- Cały pipeline i tak używa 10 bet: 4D-Humans zwraca 10, `convert_res.py` robi `shape[0][:, :10]`. Więc w ramach tej metody **nie tracicie nic** względem wersji z 300 PC.
- Realne ograniczenia są inne: model jest **męski**, `articulate.ParametricModel(use_pose_blendshape=False)` **pomija pose blendshapes**, a `models/urdf/physics.urdf` ma **stałe masy/bezwładności**. Dla kobiet, dzieci i nietypowych sylwetek to jest źródło błędu większe niż 10 PC.
- Jeśli chcecie to zmierzyć: policzcie MPJPE po `convert_res.py` dla kształtu z 10 PC vs pełnego (300 PC) na tych samych pozach, przy GT shape z datasetu.

**Krótka odpowiedź:** „skwantyzowany PhysDynPose” nie istnieje; kwantyzacja dotyczy wyłącznie zewnętrznego HMR2. fp16 jest bezpieczne, int8 wymaga pomiaru i raczej nie warto, bo zysk liczy się tylko na etapie 2, który i tak jest raz na sekwencję.

---

## 7. Czy da się to uruchomić do naszej pracy badawczej?

**Tak, warunkowo.** Potrzebne:

1. **GPU NVIDIA z CUDA** (≥11 GB dla DROID-SLAM; 4D-Humans ~8–12 GB). Bez tego etapy 2–3 odpadają, chyba że:
   - zrezygnujecie z DROID-SLAM (`--no-slam`, macie GT kamerę),
   - 4D-Humans zastąpicie własnym estymatorem SMPL (np. HMR2/SPEC/WHAM) produkującym `out_list.pkl` w tym samym formacie,
   - albo uruchomicie 4D-Humans na CPU hurtowo (setki sekund na klatkę × tysiące klatek = niepraktyczne).
2. **Licencja SMPL** — `SMPL_MALE.pkl`, `SMPL_NEUTRAL.pkl`, `SMPL_FEMALE.pkl`, `J_regressor_extra.npy` (rejestracja na smpl.is.tue.mpg.de).
3. **Kontakty** — `labels_con_sta/pred_con_gt.npy` (N,4). W repo **nie ma klasyfikatora kontaktu**; dostajecie je z archiwum albo musicie wygenerować własnym modelem.
4. **Format wejść HMR2** — `out_list.pkl`: lista słowników z `pred_smpl_params` (`betas`, `body_pose`, `global_orient` jako macierze 3×3) i `pred_cam_t_full`.
5. **Własna scena** — `asset/height_map.npy` + `models/urdf/physics.urdf` są zaszyte dla jednego pomieszczenia i jednego ciała. Dla nowej sceny trzeba wygenerować height mapę (albo wyłączyć kolizje ze sceną).

### Wydajność (CPU, 1000 klatek)

| etap | czas | uwagi |
|------|------|-------|
| align | 2.5 s | w tym One-Euro + Gaussian |
| opt (QP) | 7.3 s | ~7 ms/klatkę, 358 MB |
| convert | 0.66 s | |

Dla `seq1/31` (3333 klatki) to ~25–35 s po stronie optymalizatora. **Wąskim gardłem jest wyłącznie 4D-Humans**, nie metoda.

---

## 8. Znalezione błędy/ryzyka w wydanym kodzie

1. **`requirements.txt` jest nie do zainstalowania**: `pytorch3d==0.7.5` nie ma na PyPI, `chumpy==0.71` też nie (max 0.70, która nie działa z numpy ≥ 1.24). Poprawione w tym repo (patrz `requirements.txt`).
2. **Brak cvxopt = crash** (`KeyError: 'cvxopt'`) na klatkach, gdzie Clarabel zawodzi; cvxopt jest w wymaganiach, ale bez wyjaśnienia, że jest krytyczny.
3. **`align_orb_poses`**: stałe `0.6` i `rotation_fix` dopasowane do konkretnych danych — patrz §5.
4. **`cam_ext.npy` musi być przycięty**, podczas gdy `camera_exts_full.txt` jest przycinany w kodzie — łatwo o cichy błąd.
5. **`convert_res.py` wymaga `SMPL_NEUTRAL.pkl` i `J_regressor_extra.npy`**, mimo że ich nie używa (`orig_joints=True`).
6. **Pierwsza klatka nie przechodzi przez QP** (`optimize_frame` zwraca referencję przy `q is None`) — w wynikach jest nieciągłość na starcie; przy ewaluacji warto pominąć pierwszą klatkę.
7. **`PhysicsOptimizer` wymaga ustawienia `seq_id` przed `reset_states()`**, inaczej nie tworzy URDF-a (`AttributeError: id_robot`). API-pułapka przy użyciu klasy bezpośrednio.
8. **`run.sh` nie przekazuje urządzeń do DROID-SLAM** (`--frontend_device/--backend_device` domyślnie `cuda`); na maszynie bez CUDA trzeba wołać wrapper ręcznie.
9. **`demo.py` z 4D-Humans NIE zapisuje `out_list.pkl`** — w aktualnym `main` upstreamu (sprawdzone wyszukiwaniem kodu: 0 wystąpień `out_list`) demo zapisuje tylko rendery i ewentualne meshe. README PhysDynPose („the release wrapper uses `demo.py` because it writes `out_list.pkl` directly”) nie jest już prawdziwe. Potrzebny własny adapter/patach — patrz §11.
10. **`demo.py` nie sortuje `img_paths`** (`Path.glob` bez `sorted`), a `align_slam.py` zakłada porządek czasowy; przy wielosobowej scenie `out_list` ma wiele wpisów na klatkę, podczas gdy kod indeksuje `tracks[i]` jak jedną osobę na klatkę.

---

## 9. Rekomendacja

- Do pracy z **własnymi nagraniami z GT kamerą**: używać `--no-slam`, dostarczyć `cam_ext.npy` (przycięty), intrinsicsy, kontakty i `out_list.pkl` z HMR2.
- Do **odtwarzania wyników MoviCam**: potrzebny GPU z CUDA + licencja SMPL + archiwum Edmond; etapy 4–6 można odpalać lokalnie na CPU.
- **Kwantyzacja HMR2**: fp16 — tak; int8 — tylko po pomiarze ΔMPJPE, i tylko jeśli etap 2 jest wąskim gardłem (u Was nie będzie).
- Przed publikacją wyników na nowych danych: **zweryfikować `align_slam.align_orb_poses`** (0.6 / rotation_fix), bo to najbardziej „ukryta” stała w całym repo.

---

## 10. Inny SLAM na Radeonie?

### Co pipeline wymaga od SLAM

Tylko jednego pliku: `inputs/slam/Trajectory.txt` (albo legacy `inputs/orb_slam3/Trajectory.txt`) w formacie TUM:

```
timestamp tx ty tz qx qy qz qw     # pozycja kamery camera-to-world
```

`align_slam.read_orb_trajectory()` robi z tego extrinsic przez inwersję. Liczba wpisów musi odpowiadać liczbie klatek, a intrinsicsy są brane z `camera_exts_full.txt`.

### Co realnie działa bez CUDA

| SLAM | GPU | Uwagi |
|------|-----|-------|
| **ORB-SLAM3** | ❌ CPU | **Najlepszy wybór**: brak CUDA, format TUM, a repo ma już fallback `inputs/orb_slam3/Trajectory.txt`. Wymaga OpenCV/Eigen/Pangolin, kalibracji i – w tej scenie – maskowania człowieka. |
| **COLMAP / GLOMAP** | ❌ CPU (CUDA opcjonalnie) | Offline SfM, dobre pozy na tle, skala nieoznaczona; trzeba zamaskować osobę i przeskalować do GT. |
| OpenVINS / SVO / VINS-Mono | ❌ CPU | Wymagają IMU/stereo albo dają dryf; mono + brak IMU = problem ze skalą. |
| DROID-SLAM, DPVO, MASt3R-SLAM, ROMA | ✅ CUDA | brak portu HIP/ROCm dla ich rozszerzeń CUDA |
| PyTorch-DirectML / ROCm | – | **RX 6750 XT (gfx1031) nie jest wspierany przez oficjalne ROCm**; rozszerzenia CUDA DROID-a nie skompilują się pod HIP |

### Największy problem merytoryczny

To scena **dynamiczna**: osoba zajmuje większość kadru. Klasyczny monocular SLAM bez maskowania uzna ją za outlier albo „przyklei” trajektorię do poruszającego się ciała. Praktyczne obejścia:

1. **Maska osoby** (segmentacja 2D, np. z detektora 4D-Humans/ViTDet) i SLAM tylko na tle.
2. SLAM na klatkach z **tłem widocznym** (kadry, gdy osoba nie zasłania).
3. Skala: mono SLAM jest bezskalowy — albo alignujecie do GT (`align_orb_poses`), albo skalujecie po znanym wymiarze sceny.

**Wniosek:** na tym sprzęcie realne są klasyczne SLAM-y CPU. Ale ponieważ macie GT kamerę, **najlepszy „inny SLAM” to brak SLAM-u** (`--no-slam`). SLAM ma sens tylko jako element badania („ile tracimy bez GT?”), nie jako zależność produkcyjna.

---

## 11. Czym zastąpić 4D-Humans?

### Kontrakt wyjściowy, którego wymaga `align_slam.py`

```python
out_list[i] = {
    "pred_smpl_params": {
        "betas":         (1, 10),        # numpy / tensor
        "body_pose":     (1, 23, 3, 3),  # MACIERZE rotacji, nie axis-angle
        "global_orient": (1, 3, 3),
    },
    "pred_cam_t_full": (1, 3),           # translacja korzenia w układzie kamery [m], pełna perspektywa
}
```

Wymagania dodatkowe, wynikające z kodu:

- **dokładnie jeden wpis na klatkę**, w kolejności czasowej (`tracks[i]` = klatka `i`);
- `pred_cam_t_full` w metrach i w pełnej perspektywie (nie weak-perspective);
- bety mogą się zmieniać per klatka, ale używana jest tylko ostatnia (`last_betas`);
- kształt i tak jest później nadpisywany przez `pip_reference/.../shape.pt` w `convert_res.py`.

### Uwaga: obecny 4D-Humans nie produkuje `out_list.pkl`

Aktualny `demo.py` zapisuje tylko rendery (`{img}_{person}.png`) i opcjonalnie meshe. Potrzebny jest ~10-liniowy patch:

```python
# 1) na początku main():
import joblib
all_outputs = []

# 2) w pętli "for n in range(batch_size):" (po wyliczeniu pred_cam_t_full):
all_outputs.append({
    "pred_smpl_params": {k: v[n][None].detach().cpu().numpy()
                         for k, v in out["pred_smpl_params"].items()},
    "pred_cam_t_full": pred_cam_t_full[n][None],
})

# 3) na końcu, po pętli po obrazach:
joblib.dump(all_outputs, os.path.join(args.out_folder, "out_list.pkl"))
```

Do tego: `img_paths = sorted(...)` oraz wybór jednej osoby na klatkę (np. największy box / `person_id == 0`).

### Alternatywne estymatory SMPL

| Metoda | Sprzęt | Translacja kamery | Uwagi |
|--------|--------|-------------------|-------|
| **4D-Humans / HMR2** | CUDA (~8–12 GB) | `pred_cam_t_full` (gotowe) | oryginał; na CPU ~0.5–2 s/klatkę |
| **WHAM** | CUDA | tak (w układzie kamery) | video, temporalnie stabilny; dobry zamiennik „jeden do jednego” |
| **CameraHMR** | CUDA | tak, perspektywicznie | projektowany pod **poruszającą się kamerę** — merytorycznie najbliższy |
| **GVHMR** | CUDA | tak, world-grounded | nowszy, mocny, ale cięższy |
| **SMPLer-X** | CUDA | przez konwersję | bardzo dokładny single-image |
| **TRAM** | CUDA | tak | **odpada**: sam używa DROID-SLAM wewnątrz (CUDA) |
| **FrankMocap** | CPU ✅ | weak-perspective | najprostszy do odpalenia bez GPU; słabsza dokładność; **zintegrowany: `--estimator frankmocap`** |
| **HybrIK / VIBE / TCMR** | CPU (wolno) | brak/weak-perspective | starsze, łatwiejsze instalacyjnie |

Dla metod z kamerą weak-perspective trzeba przeliczyć `(s, tx, ty, bbox)` → `pred_cam_t_full`; gotowa funkcja to `cam_crop_to_full` z 4D-Humans — warto jej użyć zamiast pisać wzór od zera.

**Praktyczny wniosek:** zamiana jest wykonalna, bo metoda jest beztreningowa, a interfejs to jeden pickle. Na tym sprzęcie każdy estymator pojedzie na CPU „w tle” (3000 klatek × 0.5–2 s = 25–100 min), a etapy 4–6 to ~1 min. Wąskim gardłem jest wyłącznie jakość i czas estymatora, nie PhysDynPose.

Czego nadal brakuje niezależnie od wyboru modelu: **kontaktów stóp** (`labels_con_sta/pred_con_gt.npy`). W repo nie ma klasyfikatora kontaktu — to trzeba dostarczyć z datasetu albo zbudować osobno (progi na wysokość/prędkość stopy to minimum).

---

## 12. Czy „inne modele + porównanie z oryginałem” to dobry temat badawczy?

### Szczera ocena

Samo „podmieniliśmy HMR2 na X i wyszło trochę gorzej/lepiej” to **inżynieria, nie badania** — recenzent zapyta „jaki jest wkład?”. PhysDynPose to warstwa optymalizacji fizycznej: jeśli podmienicie estymator, mierzycie głównie jakość estymatora, co jest już dobrze zbadane w literaturze HMR.

Ale jest kilka ram, w których to **dobry i publikowalny** temat:

1. **Wrażliwość refine'u fizycznego na błędy wejścia (najmocniejsze).**
   Nie tylko podmiana modeli, ale **wstrzykiwanie kontrolowanego szumu**: rotacje (σ = 1°, 3°, 5°, 10°), translacja (1–10 cm), jitter czasowy, gubione klatki. Pytanie badawcze: *jak dużo błędu HMR potrafi „wchłonąć” warstwa QP i gdzie jest granica?* To realnie nowa wiedza o metodzie, a nie benchmark modeli.
2. **Ile kosztuje rezygnacja z SLAM?**
   Macie GT kamerę → możecie porównać: GT vs DROID (jeśli ktoś ma GPU) vs ORB-SLAM3 vs szum na GT. Metryki: MPJPE, dryf roota, foot sliding, penetracja podłogi. To bezpośrednio odpowiada na pytanie z §5 i jest praktycznie użyteczne dla społeczności MoviCam.
3. **Kwantyzacja/kompresja estymatora jako studium deploymentu.**
   fp32 vs fp16 vs int8, wpływ na końcową fizyczną poprawność (nie tylko na MPJPE). Ciekawe, jeśli wynikiem ma być „co najmniej jaki model wystarczy do fizyki”.
4. **Generalizacja na inne ciała i sceny.**
   Metoda używa **męskiego modelu 10-PC** i **stałego URDF** (masy/bezwładności). Zbadanie wpływu dopasowania modelu ciała do osoby (mężczyzna/kobieta/dziecko, różne sceny) to konkretna luka, której oryginalna praca nie adresuje.

### Co trzeba, żeby to było publikowalne

- **Więcej niż jedna oś zmienności**: ≥3 estymatory × ≥2 scenariusze szumu × wszystkie 8 sekwencji MoviCam.
- **Metryki fizyczne, nie tylko MPJPE**: foot sliding [cm/frame], penetracja podłogi, jitter (przyspieszenie stawów), F1 kontaktów, gładkość GRF, czas obliczeń.
- **Ablacje warstwy fizyki**: wyłączanie członów QP (kontakty, tarcie, momenty) — pokazuje, *co* daje odporność.
- **Analiza niepewności**: minimum 3 seedy szumu, przedziały ufności; inaczej różnice 1–2 mm są nieodróżnialne od szumu.
- **Uczciwe ograniczenia**: stała `0.6`, `height_map`, męski model, brak klasyfikatora kontaktu.

### Ryzyka

- Odtworzenie oryginalnych liczb wymaga GPU + licencji SMPL + archiwum Edmond; bez tego recenzent zapyta o baseline.
- Wydany kod ma stałe dopasowane do jednej sceny — „generalizacja” może okazać się niemożliwa bez przepisania fragmentów (height mapa, URDF).
- Temat „benchmark estymatorów” jest zatłoczony; wartość jest w **analizie interakcji estymator ↔ warstwa fizyczna**, nie w rankingu modeli.

### Rekomendacja

Jeśli celem jest publikacja: **rama 1 + 2** (wrażliwość + koszt braku SLAM), z podmianą modeli jako jednym z eksperymentów, a nie całością pracy. Jeśli celem jest szybki projekt zaliczeniowy/raport: obecny „swap + porównanie” wystarczy, ale trzeba jasno napisać, że to walidacja inżynierska, a nie nowa metoda.

---

## 13. FrankMocap — uruchomione i zintegrowane

### Co zostało dodane

| plik | rola |
|------|------|
| [`tools/frankmocap_to_out_list.py`](C:/Github/PhysDynPose/tools/frankmocap_to_out_list.py) | uruchamia FrankMocap body na folderze klatek i zapisuje `out_list.pkl` w formacie wymaganym przez `align_slam.py` |
| [`tools/frankmocap_cpu.patch`](C:/Github/PhysDynPose/tools/frankmocap_cpu.patch) | 3 poprawki FrankMocapa potrzebne do pracy bez CUDA |
| [`run.sh`](C:/Github/PhysDynPose/run.sh) | nowy przełącznik `--estimator 4dhumans|frankmocap` (domyślnie `4dhumans`, zachowanie bez zmian) |

### Instalacja FrankMocapa (CPU, WSL)

```bash
conda create -y -n frankmocap -c conda-forge --override-channels python=3.8 pip
conda activate frankmocap
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==1.23.5 opencv-python-headless smplx easydict pyyaml tqdm scipy pillow torchgeometry==0.1.2 joblib

git clone --depth 1 https://github.com/facebookresearch/frankmocap.git
cd frankmocap

# 2D pose detector (wymagany przez FrankMocap)
git clone --depth 1 https://github.com/jhugestar/lightweight-human-pose-estimation.pytorch.git \
    detectors/body_pose_estimator
curl -L -o extra_data/body_module/body_pose_estimator/checkpoint_iter_370000.pth \
    https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth

# regressor + dane SPIN
curl -L -o extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt \
    https://dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt
curl -L -o /tmp/spin_data.bin http://visiondata.cis.upenn.edu/spin/data.tar.gz
tar -xf /tmp/spin_data.bin -C extra_data/body_module   # UWAGA: to nie jest gzip, mimo .tar.gz
mv extra_data/body_module/data extra_data/body_module/data_from_spin

# model SMPL (licencja) -> extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
git apply /path/to/PhysDynPose/tools/frankmocap_cpu.patch
```

### Poprawki CPU (zawarte w patchu)

1. `bodymocap/body_bbox_detector.py`: `net.cuda()` → tylko gdy CUDA jest dostępne.
2. `bodymocap/body_bbox_detector.py`: `__infer_fast(..., cpu=not torch.cuda.is_available())` — wcześniej twarde `tensor_img.cuda()`.
3. `bodymocap/body_mocap_api.py`: `torch.load(..., map_location="cpu")` oraz `.cuda()` → `.to(pred_rotmat.device)`.

Bez tych trzech zmian FrankMocap wywala się na maszynie bez NVIDIA już przy ładowaniu checkpointu.

### Jak to podłączyć

```bash
FRANKMOCAP_ROOT=~/frankmocap \
FRANKMOCAP_PYTHON=~/miniconda3/envs/frankmocap/bin/python \
./run.sh --seq_no 1 --surface 31 --frames_dir /path/to/frames --no-slam --estimator frankmocap
```

Adapter sam odczytuje `fx` z `camera_exts_full.txt` (`--frame_index` = pierwsza klatka zakresu), bo od ogniskowej zależy metryczna skala `pred_cam_t_full`. Alternatywnie `--focal_px`. Można też podać stały `--bbox x y w h` i pominąć detektor.

### Co dokładnie robi adapter

1. detekcja 2D (lightweight pose) → wybór **największej** osoby (pipeline obsługuje jedną osobę),
2. HMR (EFT) → `pred_rotmat (1,24,3,3)`, `pred_betas (1,10)`, `pred_camera = (s, tx, ty)`,
3. konwersja weak-perspective → pełna perspektywa wzorem z 4D-Humans (`cam_crop_to_full`),
4. zapis `body_pose = rotmat[:,1:]`, `global_orient = rotmat[:,0]`, `pred_cam_t_full`.

Brak detekcji w klatce → powtórzenie ostatniej predykcji (z licznikiem ostrzeżeń), żeby zachować jeden wpis na klatkę.

### Wynik uruchomienia (20 klatek, CPU)

```
FrankMocap: 3.1 s  (6.4 fps, 0 klatek bez detekcji)
align_slam: 0.26 s
run_opt:    0.47 s
convert:    0.17 s
tensor wyjściowy: joints (20,24,3), finite=True, root y std = 6 mm
```

Cały łańcuch `run.sh --estimator frankmocap --no-slam` przechodzi bez błędów. **Uwaga:** użyty model SMPL był syntetyczny (brak licencji), więc liczby nie mają znaczenia fizycznego — walidowany jest przepływ danych i wydajność, nie jakość.

### Ograniczenia FrankMocapa w tym workflow

- **Brak trackingu** — predykcja per klatka; Tożsamość osoby wynika z wyboru największego boxa. Przy przysłonięciach mogą wystąpić przeskoki; 4D-Humans `track.py` (PHALP) jest stabilniejszy.
- **Dokładność** — model z 2020 r. (EFT/HMR) jest wyraźnie słabszy od HMR2.0b, szczególnie dla nietypowych póz i głębokości.
- **Jedna osoba** — zgodnie z ograniczeniem pipeline'u.
- Wymaga **jawnej ogniskowej** (inaczej skala świata jest błędna); adapter bierze ją z intrinsicsów, ale przy kamerze nieznanej trzeba ją znać.

---

## 14. Co pobrać ze strony SMPL

Ze strony [smpl.is.tue.mpg.de/download.php](https://smpl.is.tue.mpg.de/download.php) potrzebne są **dwa archiwa** (reszta — UV, DMPL, FBX, Maya, Unity, USMPL, SMPL-X — nie jest do niczego potrzebna):

| # | Archiwum | Po co |
|---|----------|-------|
| 1 | **`SMPL_python_v.1.1.0.zip`** (female/male/neutral, 300 PC) | obowiązkowe: `data/body_models/smpl/SMPL_*.pkl` oraz modele dla FrankMocapa / 4D-Humans |
| 2 | **`SMPL_python_v.1.0.0.zip`** (female/male, 10 PC) | zalecane: dokładnie `models/SMPL_male_10PCs.pkl` (10 współczynników, męski) |

### Gdzie co trafia

| plik z archiwum | miejsce docelowe |
|-----------------|------------------|
| v1.0.0 `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` | `models/SMPL_male_10PCs.pkl` |
| v1.1.0 `.../basicmodel_m_lbs_10_207_0_v1.1.0.pkl` | `data/body_models/smpl/SMPL_MALE.pkl` |
| v1.1.0 `.../basicmodel_f_lbs_10_207_0_v1.1.0.pkl` | `data/body_models/smpl/SMPL_FEMALE.pkl` |
| v1.1.0 `.../basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` | `data/body_models/smpl/SMPL_NEUTRAL.pkl` |
| v1.1.0 neutral (kopia) | `<FrankMocap>/extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` |
| v1.1.0 neutral (kopia) | `<4D-Humans>/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` |
| `J_regressor_extra.npy` | `data/J_regressor_extra.npy` — **nie ma go na stronie SMPL**; pochodzi z SPIN (jest w `~/frankmocap/extra_data/body_module/data_from_spin/J_regressor_extra.npy`) |

Uwaga: **v1.0.0 zawiera tylko model żeński i męski — bez neutralnego** (sprawdzone w archiwum). Neutral pochodzi z v1.1.0.

Nazwa `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` jest zaszyta w 4D-Humans i FrankMocapie, ale **liczy się zawartość, nie nazwa** — można skopiować `SMPL_NEUTRAL.pkl` pod tę nazwę (smplx i tak przytnie 300 PC do 10 bet).

### Instalacja z gotowych zipów (wykonana)

```bash
python tools/install_smpl_models.py \
    --v100 SMPL_python_v.1.0.0.zip \
    --v110 SMPL_python_v.1.1.0.zip \
    --frankmocap ~/frankmocap
```

Skrypt [`tools/install_smpl_models.py`](C:/Github/PhysDynPose/tools/install_smpl_models.py) robi wszystko: rozpakowuje tylko potrzebne pliki, konwertuje chumpy → numpy, przycina `shapedirs` do 10 PC, kopiuje `J_regressor_extra.npy`, instaluje neutralny model w FrankMocapie i weryfikuje każdy plik przez `smplx`.

Wynik u nas:

```
models/SMPL_male_10PCs.pkl          39.1 MB  betas=10
data/body_models/smpl/SMPL_MALE.pkl 39.1 MB  betas=10
.../SMPL_FEMALE.pkl                 39.1 MB  betas=10
.../SMPL_NEUTRAL.pkl                39.0 MB  betas=10
data/J_regressor_extra.npy           1.3 MB
FrankMocap .../basicModel_neutral... 39.0 MB betas=10

verified SMPL_male_10PCs: joints=45, pelvis=[-0.002, -0.241, 0.029]
verified SMPL_NEUTRAL:    joints=45, pelvis=[-0.002, -0.223, 0.028]
```

Po instalacji pełny łańcuch `run.sh --estimator frankmocap --no-slam` przeszedł na prawdziwych modelach: tensory skończone, `joints (20,24,3)`, root y std ≈ 6 mm.

Archiwa `.zip` są w `.gitignore` (`SMPL_python_v.*.zip`) — można je usunąć po instalacji (~400 MB).


### Dlaczego optimizer wymaga 10 PC

`articulate.ParametricModel` robi `torch.tensordot(shape(10), shapedirs(..., K))`, więc dla `models/SMPL_male_10PCs.pkl` **K musi wynosić dokładnie 10**. Plik v1.1.0 ma 300 — trzeba przyciąć:

```python
import pickle
import numpy as np

with open("SMPL_MALE.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")
data["shapedirs"] = np.asarray(data["shapedirs"])[:, :, :10]
with open("models/SMPL_male_10PCs.pkl", "wb") as f:
    pickle.dump(data, f)
```

To nie jest identyczne z oryginalnym v1.0.0 (v1.1.0 ma zaktualizowany template), dlatego do odtwarzania liczb z pracy lepiej pobrać v1.0.0.

### Chumpy

Pliki SMPL v1.0.0 to pikle **chumpy**, a `import chumpy` wywala się na numpy ≥ 1.24 (`from numpy import bool`). Rozwiązania:

1. przekonwertować raz w envie z numpy 1.23 (np. `frankmocap`, `pip install --no-build-isolation chumpy==0.70`) i zapisać ponownie jako czyste numpy — potem env PhysDynPose nie potrzebuje chumpy;
2. albo przed importem ustawić aliasy:
   ```python
   import numpy as np
   np.bool = bool; np.int = int; np.float = float
   np.complex = complex; np.object = object; np.str = str
   import chumpy  # teraz zadziała
   ```

### Licencja

SMPL jest **tylko do badań niekomercyjnych**. Nie commituj tych plików — `.gitignore` już obejmuje `models/SMPL_male_10PCs.pkl`, `data/body_models/` i `data/J_regressor_extra.npy`.
