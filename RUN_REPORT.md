# PhysDynPose — pierwsze uruchomienie i pomiary (etap 1)

| | |
|---|---|
| **Metoda / dane** | PhysDynPose (CVPRW 2025), zbiór MoviCam |
| **Stanowisko** | Windows 11 + WSL2 (Ubuntu), Ryzen 8 wątków, GPU **AMD Radeon RX 6750 XT** (brak CUDA), 16 GB RAM |
| **Kod** | commit `97569d8` |
| **Dane** | wejścia syntetyczne w formacie MoviCam — zbiór nie był jeszcze pobrany |
| **Data** | 20.10.2026 |
| **Etap 2** | [RUN_REPORT_DAMSI_GPU.md](RUN_REPORT_DAMSI_GPU.md) — pełny pipeline na klastrze GPU i oryginalnych danych |

## 1. Cel

Sprawdziłem, które etapy metody dają się uruchomić na dostępnym sprzęcie, ile kosztują
obliczeniowo i jakie przeszkody trzeba pokonać w wydanym kodzie. Na tej stacji nie ma CUDA,
a zbiór MoviCam nie był wtedy pobrany, więc test wykonałem na wejściach syntetycznych —
mierzy on **koszt kodu, nie jakość metody**.

## 2. Środowisko: co nie działało i jak to rozwiązałem

Repozytorium zakłada Linuksa z CUDA, a stacja ma Windowsa i kartę AMD, więc całość postawiłem
w WSL2 (Ubuntu) z Minicondą w katalogu domowym.

| Problem | Objaw | Rozwiązanie |
|---|---|---|
| Python 3.13 na Windows | `torch==2.1.0` i `pytorch3d` nie mają wheeli dla cp313, pytorch3d nie ma wheeli Windows w ogóle | WSL2 + Miniconda + Python 3.10 |
| brak `pip`/`venv` w WSL | `ensurepip is not available` | Miniconda instalowana do `$HOME`, bez `sudo` |
| nowa conda wymaga akceptacji ToS kanałów Anacondy | `CondaToSNonInteractiveError` | `-c conda-forge --override-channels` |
| `pybullet==3.2.5` | brak wheela, kompilacja wymaga `g++` | `conda install -c conda-forge pybullet` |
| `cvxopt==1.2.6` | kompilacja wymaga BLAS/LAPACK | `conda install -c conda-forge cvxopt` (1.3.3) |
| RBDL (bindings Pythona) | nie ma na PyPI, pakiet conda-forge `rbdl` to tylko C++ | build ze źródeł (`v3.3.1`, `-DRBDL_BUILD_PYTHON_WRAPPER=ON`), `rbdl.so` do `site-packages` |
| `pytorch3d==0.7.5` | **nie istnieje na PyPI** | w teście zastąpiony 20-liniowym shimem dwóch funkcji; docelowo build ze źródeł na maszynie z CUDA |
| `chumpy==0.71` | **nie istnieje na PyPI** (max 0.70), a 0.70 nie importuje się z numpy ≥ 1.24 (`from numpy import bool`) | konwersja pickli SMPL do czystego numpy (patrz §2.2) |

Działające wersje: Python 3.10.22, torch 2.1.0+cpu, numpy 1.26.4, scipy 1.10.1, smplx 0.1.28,
pybullet 3.2.5, qpsolvers 4.13 + clarabel 0.11.1, cvxopt 1.3.3, RBDL 3.3.1 (zbudowany lokalnie).

### 2.1 Solver QP: cvxopt jest obowiązkowy

`dynamics_scene.py` woła najpierw **Clarabel**, a przy porażce **cvxopt**. Na części klatek
(wynikających np. z nóg poniżej podłogi) Clarabel zwraca `None` i bez cvxopt kod kończy się
`KeyError: 'cvxopt'`. To nie jest zależność „nice to have” — bez cvxopt optymalizacja pada.

### 2.2 Pliki modelu ciała

Modele SMPL wymagają rejestracji na smpl.is.tue.mpg.de i są licencjonowane wyłącznie do badań
niekomercyjnych. Potrzebne są dwa archiwa: `SMPL_python_v.1.1.0.zip` (wersje female/male/neutral,
300 PC) oraz `SMPL_python_v.1.0.0.zip`, z którego pochodzi dokładnie `models/SMPL_male_10PCs.pkl`.
Do tego `J_regressor_extra.npy`, którego na stronie SMPL nie ma — jest w pakiecie SPIN.

Instaluję je skryptem [tools/install_smpl_models.py](tools/install_smpl_models.py), który rozpakowuje
tylko potrzebne pliki, konwertuje chumpy → numpy, przycina `shapedirs` do 10 PC, kopiuje
`J_regressor_extra.npy`, instaluje model neutralny dla FrankMocapa i weryfikuje każdy plik przez
`smplx`:

```bash
python tools/install_smpl_models.py --v100 SMPL_python_v.1.0.0.zip \
    --v110 SMPL_python_v.1.1.0.zip --frankmocap ~/frankmocap
```

Dlaczego akurat 10 PC: `articulate.ParametricModel` liczy
`torch.tensordot(shape(10), shapedirs(..., K))`, więc **K musi wynosić dokładnie 10** — plik
z 300 PC trzeba przyciąć. Przycięcie v1.1.0 nie jest równoważne oryginałowi v1.0.0 (v1.1.0 ma
zaktualizowany template), dlatego do odtwarzania liczb z pracy lepiej użyć v1.0.0.

## 3. Dane testowe

Zbioru nie miałem, a modeli SMPL nie da się legalnie pobrać bez rejestracji, więc żeby w ogóle
przejść przez kod, wygenerowałem wejścia sam:

1. **syntetyczny model w formacie SMPL** — 24 stawy, 6890 wierzchołków, poprawna hierarchia,
   miednica w początku układu; spełnia wymagania `smplx` i `articulate.ParametricModel`,
   ale geometria jest losowa;
2. pliki wejściowe w **formacie MoviCam**: `camera_exts_full.txt` (bloki 3-liniowe),
   `cam_ext.npy` (N,4,4), `pred_con_gt.npy` (N,4), `out_list.pkl` w formacie 4D-Humans,
   `pip_reference/.../shape.pt`;
3. przepuściłem to przez `align_slam.py`, `run_opt.py`, `convert_res.py`.

Wszystkie trzy etapy kończą się sukcesem, tensory są skończone, QP się rozwiązuje. **Liczby nie
mają sensu fizycznego** — to test ścieżek kodu i wydajności.

Dwie pułapki, na które się przy tym natknąłem:

- `align_slam.py` przycina intrinsicsy do zakresu z `config.FRAME_RANGES`, ale **`cam_ext.npy`
  musi być już przycięty** do tego samego zakresu. Pełny plik powoduje, że
  `min(len(intrinsics), len(cam_ext), len(tracks))` cicho obcina złe klatki.
- `convert_res.py` ładuje `data/body_models/smpl/SMPL_NEUTRAL.pkl` **oraz**
  `data/J_regressor_extra.npy`, mimo że woła `orig_joints=True` i tego regressora nie używa.
  Brak pliku kończy się `AssertionError`.

## 4. Pomiary (CPU, 1000 klatek)

| Etap | Czas | Uwagi |
|---|---|---|
| `align_slam.py` | 2,5 s | w tym filtr One-Euro i Gaussian |
| `run_opt.py` (QP) | 7,3 s | ~7 ms/klatkę, 358 MB RSS |
| `convert_res.py` | 0,66 s | głównie import `smplx` |

Dla `seq1/31` (3333 klatki) daje to ~25–35 s po stronie optymalizatora. Cały wkład metody jest
więc obliczeniowo pomijalny — wąskim gardłem jest wyłącznie estymator pozy.

## 5. SLAM czy kamera z ground truth

Prześledziłem ścieżkę `--slam` i okazało się, że **nie jest samodzielna**: `align_orb_poses()`
skaluje trajektorię SLAM względem GT (`gt_scale / orb_scale`) i zakotwicza ją w pierwszej pozycji
GT. SLAM służy tam do wygładzenia toru kamery, a nie do wyznaczenia skali, i zawiera dwie stałe
dopasowane do konkretnej konwencji danych:

```python
scale = gt_scale / orb_scale * 0.6                      # współczynnik skali
rotation_fix = np.array([[0,0,1],[0,-1,0],[1,0,0]])     # zamiana osi
```

Sprawdziłem to doświadczalnie — podałem `align_slam.py --slam` trajektorię **identyczną z GT**
(czyli pozornie idealny SLAM):

```
GT displacement        1.998 m
"SLAM" displacement    1.199 m      ← dokładnie ×0.6
rotation diff          0.0°
różnica translacji korzenia po optymalizacji: średnio 1.17 m, maks. 2.32 m
```

Wniosek: przy własnej kamerze z GT trzeba używać `--no-slam`. Ścieżka `--slam` wprowadza błąd
skali 40 % i obrót osi; na nowych danych wymaga rekalibracji. Etap 2 potwierdził to liczbowo
(3,4× mniejszy błąd w układzie świata).

Osobna uwaga merytoryczna: MoviCam to scena **dynamiczna** — osoba zajmuje większość kadru, więc
monokularowy SLAM bez maskowania albo uzna ją za outlier, albo „przyklei” trajektorię do
poruszającego się ciała. To kolejny argument, żeby nie opierać metody na SLAM-ie, jeśli jest GT.

## 6. Obejście braku CUDA: FrankMocap

Ponieważ 4D-Humans na CPU jest niepraktyczny, podłączyłem jako alternatywny estymator
**FrankMocap** (CPU, 6–7 fps). Co dodałem:

| Plik | Rola |
|---|---|
| [tools/frankmocap_to_out_list.py](tools/frankmocap_to_out_list.py) | uruchamia FrankMocap body na folderze klatek i zapisuje `out_list.pkl` w formacie wymaganym przez `align_slam.py` |
| [tools/frankmocap_cpu.patch](tools/frankmocap_cpu.patch) | trzy poprawki potrzebne do pracy bez CUDA |
| [run.sh](run.sh) | przełącznik `--estimator 4dhumans\|frankmocap`, domyślnie `4dhumans` (zachowanie bez zmian) |

Poprawki w patchu:

1. `bodymocap/body_bbox_detector.py`: `net.cuda()` wywoływane tylko przy dostępnym CUDA;
2. `bodymocap/body_bbox_detector.py`: `__infer_fast(..., cpu=not torch.cuda.is_available())`
   zamiast twardego `tensor_img.cuda()`;
3. `bodymocap/body_mocap_api.py`: `torch.load(..., map_location="cpu")` oraz `.cuda()` →
   `.to(pred_rotmat.device)`.

Bez tych zmian FrankMocap wywala się na maszynie bez NVIDIA już przy ładowaniu checkpointu.

Adapter robi cztery rzeczy: detekcja 2D i wybór największej osoby (pipeline obsługuje jedną),
regresja EFT/HMR → `pred_rotmat`, `pred_betas`, `pred_camera`, konwersja weak-perspective →
pełna perspektywa wzorem `cam_crop_to_full` z 4D-Humans, zapis `out_list.pkl`. Ogniskową bierze
z `camera_exts_full.txt` (albo z `--focal_px`), bo od niej zależy metryczna skala translacji.
Brak detekcji w klatce → powtórzenie ostatniej predykcji, żeby zachować jeden wpis na klatkę.

Pomiar na 20 klatkach, CPU:

```
FrankMocap: 3.1 s  (6.4 fps, 0 klatek bez detekcji)
align_slam: 0.26 s
run_opt:    0.47 s
convert:    0.17 s
wyjście:    joints (20,24,3), finite=True, root y std = 6 mm
```

Cały łańcuch `run.sh --estimator frankmocap --no-slam` przechodzi bez błędów. Wynik liczbowy
nic nie mówi o jakości (model SMPL był wtedy syntetyczny) — walidowany jest przepływ danych
i wydajność.

Ograniczenia tego obejścia: brak trackingu (predykcja per klatka, tożsamość z największego boxa,
możliwe przeskoki przy przysłonięciach), wyraźnie słabsza dokładność od HMR2.0, jedna osoba,
wymagana znana ogniskowa.

## 7. Błędy i ryzyka w wydanym kodzie

1. **`requirements.txt` jest nie do zainstalowania** — `pytorch3d==0.7.5` i `chumpy==0.71` nie
   istnieją na PyPI (poprawiłem opis w tym pliku).
2. **Brak cvxopt = crash** (`KeyError: 'cvxopt'`) na klatkach, gdzie Clarabel zawodzi; w
   wymaganiach nie ma informacji, że jest krytyczny.
3. **`align_orb_poses`**: stałe `0.6` i `rotation_fix` dopasowane do jednej konwencji danych (§5).
4. **`cam_ext.npy` musi być przycięty**, podczas gdy `camera_exts_full.txt` jest przycinany
   w kodzie — łatwo o cichy błąd.
5. **`convert_res.py` wymaga `SMPL_NEUTRAL.pkl` i `J_regressor_extra.npy`**, mimo że ich nie używa.
6. **Pierwsza klatka nie przechodzi przez QP** (`optimize_frame` zwraca referencję przy
   `q is None`) — w wyniku jest nieciągłość na starcie; przy ewaluacji warto pominąć pierwszą klatkę.
7. **`PhysicsOptimizer` wymaga ustawienia `seq_id` przed `reset_states()`**, inaczej nie tworzy
   URDF-a (`AttributeError: id_robot`) — pułapka przy użyciu klasy bezpośrednio.
8. **`run.sh` nie przekazuje urządzeń do DROID-SLAM** (`--frontend_device`/`--backend_device`
   domyślnie `cuda`); bez CUDA trzeba wołać wrapper ręcznie.
9. **`demo.py` z 4D-Humans nie zapisuje `out_list.pkl`** — w obecnym `main` upstreamu (0 wystąpień
   `out_list`) demo zapisuje tylko rendery i ewentualne meshe, wbrew temu, co twierdzi README
   PhysDynPose. Potrzebny własny patch.
10. **`demo.py` nie sortuje `img_paths`** (`Path.glob` bez `sorted`), a `align_slam.py` zakłada
    porządek czasowy; przy scenie wielosobowej `out_list` ma wiele wpisów na klatkę, podczas gdy
    kod indeksuje `tracks[i]` jak jedną osobę na klatkę.

Etap 2 (klaster GPU, oryginalne dane) potwierdził punkty 1, 3, 4, 9, 10 i dodał trzy błędy
blokujące:

11. **Wideo z releasu jest już przycięte** do zakresu z `config.FRAME_RANGES`, czego
    `extract_movicam_frames.py` nie zakłada — zapisuje 748 z 2324 klatek i kończy wyjątkiem.
12. **`run_droid_slam.py` nie ustawia `args.upsample`** wymaganego przez DROID-SLAM
    (`AttributeError`).
13. **Skala translacji z 4D-Humans jest ~33× za duża**, bo `cam_crop_to_full` używa domyślnej
    ogniskowej konfiguracji demo (~42 857 px) zamiast rzeczywistej (1289,9 px). Efektem jest
    nieosiągalne zadanie QP i `RuntimeError: Quadratic program failed during optimization`
    na pierwszej klatce, czyli całkowita blokada etapu optymalizacji. Poprawka:
    [tools/fix_out_list_focal.py](tools/fix_out_list_focal.py).

Szczegóły i pomiary dla punktów 11–13: [RUN_REPORT_DAMSI_GPU.md](RUN_REPORT_DAMSI_GPU.md) §7.

## 8. Wnioski z etapu 1

1. **Warstwa fizyczna, czyli wkład własny metody, jest obliczeniowo pomijalna** — 7 ms/klatkę,
   ~30 s na 3333 klatki. Cały koszt leży po stronie estymatora pozy, więc eksperymenty po stronie
   optymalizacji można liczyć bez ograniczeń.
2. **Bez CUDA da się przejść cały łańcuch tylko z obejściem estymatora** — FrankMocap na CPU
   działa (6–7 fps), ale kosztem dokładności; produkcyjnie potrzebny jest GPU.
3. **Ścieżka `--slam` nie nadaje się do własnych danych** bez rekalibracji stałych — przy kamerze
   z GT należy używać `--no-slam`. Potwierdzone doświadczalnie na trajektorii identycznej z GT.
4. **Bariera wejścia jest wysoka**: osiem problemów instalacyjnych (§2), licencyjne pliki SMPL
   (§2.2) i trzynaście błędów w wydanym kodzie (§7). To wyjaśnia, dlaczego wyniki z pracy trudno
   odtworzyć bez własnych poprawek.

Pomiary z tego etapu dotyczą danych syntetycznych. Uruchomienie na oryginalnym zbiorze
i pomiary jakości znajdują się w [RUN_REPORT_DAMSI_GPU.md](RUN_REPORT_DAMSI_GPU.md).
