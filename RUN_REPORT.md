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

---

## 9. Rekomendacja

- Do pracy z **własnymi nagraniami z GT kamerą**: używać `--no-slam`, dostarczyć `cam_ext.npy` (przycięty), intrinsicsy, kontakty i `out_list.pkl` z HMR2.
- Do **odtwarzania wyników MoviCam**: potrzebny GPU z CUDA + licencja SMPL + archiwum Edmond; etapy 4–6 można odpalać lokalnie na CPU.
- **Kwantyzacja HMR2**: fp16 — tak; int8 — tylko po pomiarze ΔMPJPE, i tylko jeśli etap 2 jest wąskim gardłem (u Was nie będzie).
- Przed publikacją wyników na nowych danych: **zweryfikować `align_slam.align_orb_poses`** (0.6 / rotation_fix), bo to najbardziej „ukryta” stała w całym repo.
