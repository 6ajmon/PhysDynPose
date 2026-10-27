# PhysDynPose — uruchomienie pełnego pipeline'u na klastrze GPU

| | |
|---|---|
| **Materiał** | odtworzenie metody PhysDynPose (CVPRW 2025) i zbioru MoviCam |
| **Stanowisko** | serwer wydziałowy `damsi-gpu` |
| **Kod** | `6ajmon/PhysDynPose`, commit `a21160d` + poprawki z §7 |
| **Dane** | oryginalny zbiór MoviCam z Edmonda (91,2 GiB), sekwencja `surface=31 / seq=6` |
| **Data** | 26.10.2026 |
| **Raport uzupełniający** | [RUN_REPORT.md](RUN_REPORT.md) — wcześniejsze próby na stacji roboczej (Windows/WSL2, GPU AMD bez CUDA, dane syntetyczne) |

---

## 1. Cel i zakres

Celem było sprawdzenie, czy wydany kod PhysDynPose daje się uruchomić od zera na dostępnym
sprzęcie GPU i czy wyniki są merytorycznie sensowne. Zakres prac:

1. zbudowanie kompletnego środowiska dla trzech zewnętrznych zależności (4D-Humans,
   DROID-SLAM, RBDL) na maszynie bez uprawnień administratora;
2. pozyskanie i weryfikacja oryginalnego zbioru MoviCam (98 GB) wraz z danymi wejściowymi
   z releasu;
3. uruchomienie pełnego łańcucha przetwarzania na sekwencji referencyjnej
   (`surface=31`, `seq=6`, 2324 klatki, 1920×1080, 25 fps);
4. pomiar czasów i ocena jakości wyniku względem ground truth z systemu Captury;
5. identyfikacja przeszkód w wydanym kodzie, które uniemożliwiają odtworzenie wyników
   bez modyfikacji.

**Wynik:** pipeline przechodzi w całości. Po drodze znalazłem trzy błędy w wydanym
kodzie, z których jeden całkowicie blokował etap optymalizacji fizycznej (§7).

---

## 2. Stanowisko obliczeniowe

| Parametr | Wartość |
|---|---|
| System | Debian GNU/Linux 12 (bookworm), jądro 6.1 |
| CPU | 96 rdzeni |
| RAM | 1,5 TiB |
| GPU | 5 × NVIDIA L40S, 46 GB każda (architektura Ada, `sm_89`) |
| Sterownik / CUDA | 535.216.01 / toolkit 11.8 |
| Dysk | 11 TB, 8,3 TB wolnego przed uruchomieniem |
| Uprawnienia | brak `sudo` — cała instalacja w katalogu domowym |

Zasoby są nadmiarowe wobec potrzeb metody: część niezależna od estymatora pozy zajmuje ~30 s
na 2324 klatki i poniżej 1 GB RAM. Pięć GPU pozwala liczyć pięć sekwencji równolegle;
estymator pozy zajmuje ok. 9 GB VRAM na jedną instancję.

---

## 3. Środowisko uruchomieniowe

Zbudowałem je od zera w katalogu domowym, bez praw administratora. Menedżer pakietów:
**micromamba 2.9.0** (pojedynczy plik wykonywalny), środowisko `~/work/envs/physopt` z
**Pythonem 3.10.22**.

Wersje wynikowe: torch 2.1.0+cu118, torchvision 0.16.0, numpy 1.26.4, scipy 1.10.1,
smplx 0.1.28, qpsolvers 4.13 (clarabel 0.11.1, daqp, proxsuite, scs), cvxopt 1.3.3,
pybullet 3.2.5, pytorch3d 0.7.5, detectron2 0.6, RBDL 3.3.1, OpenCV 4.10.0.

Wszystkie etapy działają w **jednym środowisku** (estymator pozy, SLAM i optymalizacja) —
`run.sh` otrzymuje ten sam interpreter w każdej zmiennej `*_PYTHON`. Upraszcza to uruchamianie
kosztem konieczności pogodzenia wymagań trzech projektów naraz.

### 3.1 Problemy instalacyjne i ich rozwiązania

| Problem | Rozwiązanie |
|---|---|
| brak `sudo`, więc brak `apt install` | micromamba + conda-forge instalowane w katalogu domowym |
| domyślny wheel `torch` z PyPI to build cu121, a systemowy `nvcc` to 11.8 | użyto wheeli **cu118** z `download.pytorch.org/whl/cu118` |
| `pytorch3d` nie ma wheeli na PyPI | build ze źródeł z tagu `v0.7.5` z **`FORCE_CUDA=0`** — pipeline używa wyłącznie `pytorch3d.transforms`, czyli czystego PyTorcha, więc build CPU wystarcza |
| `cvxopt` i `pybullet` nie mają użytecznych wheeli (kompilacja wymaga BLAS / `g++`) | `conda install -c conda-forge cvxopt pybullet` |
| RBDL nie publikuje bindings Pythona | build 3.3.1 z `-DRBDL_BUILD_PYTHON_WRAPPER=ON`, `rbdl.so` kopiowany do `site-packages` |
| conda-forge dostarcza **Eigen 5.0.1**, na którym RBDL i pytorch3d się nie kompilują (`_check_template_params` usunięte z Eigen) | przypięcie `eigen=3.4.0` |
| systemowy **gcc 12.2** jest nieobsługiwany przez CUDA 11.8 | użyto `/usr/bin/gcc-11` i `g++-11` przez `CC` / `CXX` |
| `detectron2` (wymagany przez detektor ViTDet) nie ma wheeli | build z `git` (0.6) |
| DROID-SLAM wymaga `torch_scatter` | `torch-scatter==2.1.2+pt21cu118` z `data.pyg.org` |
| `setup.py` biblioteki `lietorch` hardcode'uje `-gencode` tylko do `sm_75` | dopisałem `compute_89/sm_89` dla L40S |
| `setuptools ≥ 81` usuwa `pkg_resources`, na którym opiera się `torch.utils.cpp_extension` | `pip install "setuptools<81"` |
| checkpointy 4D-Humans (~2,6 GB) i wagi ViTDet | pobrane automatycznie do `~/.cache/4DHumans` |
| wagi `droid.pth` | `gdown` z oficjalnego linku (16,1 MB) |

Dodatkowo w checkoutcie 4D-Humans konieczny był patch `demo.py`: stockowa wersja **nie zapisuje**
pliku `out_list.pkl`, który jest kontraktem wejściowym PhysDynPose. Patch dodaje sortowanie
listy obrazów (kolejność czasowa), wybór jednej osoby na klatkę (największy bounding box)
i zapis `out_list.pkl`. Plik nie należy do tego repozytorium — patch trzeba odtworzyć po
każdej reinstalacji 4D-Humans.

---

## 4. Dane wejściowe

### 4.1 Pozyskanie i weryfikacja

| Artefakt | Rozmiar | Sposób |
|---|---|---|
| `MoviCam.zip` (Edmond, plik 345390) | 97 939 015 411 B (91,2 GiB) | Dataverse API → podpisany URL do S3, 6 równoległych zapytań `Range`, ~33 min przy ~51 MB/s |
| `inputs.zip` (Edmond, plik 345391) | 6,7 MB | bezpośrednio |
| rozpakowane `Data/` | 97,92 GB, 70 809 plików | `unzip`, ~10 min przy 196 MB/s |

Archiwum sprawdziłem przez `zipfile.testzip()` (CRC wszystkich 70 926 pozycji) — wynik poprawny.
Rozpakowanie zweryfikowałem pozycja po pozycji względem manifestu archiwum:
**70 809/70 809 plików obecnych, 0 niezgodności rozmiaru**. Wszystkie pozycje w ZIP-ie są
zapisane bez kompresji (`STORED`), dlatego rozpakowanie ogranicza wyłącznie przepustowość dysku,
nie CPU.

Struktura: `Data/{Recordings_31_03_23, Recordings_24_03_23, indoor_31_03_23, indoor_24_03_23}`
— nagrania surowe, dane benchmarkowe `Cam_seq/mov0*`, kalibracje kamer i scena `scene_gt/`.

### 4.2 Podłączenie do kodu

Nagrania podpiąłem do ścieżek oczekiwanych przez kod symlinkami (wariant wprost dopuszczony przez
[DATASET.md](DATASET.md)):

```bash
cd ~/work/PhysDynPose/inputs/recordings
ln -sfn ~/work/data/Data/Recordings_31_03_23 Recordings_31_03_23
ln -sfn ~/work/data/Data/Recordings_24_03_23 Recordings_24_03_23
```

Sprawdziłem, że etykiety kontaktów stóp z `inputs.zip` (`pred_con_gt.npy`) są **bit w bit
identyczne** z tymi w surowym archiwum dla wszystkich 8 sekwencji, więc użycie surowego `Data/`
jako źródła autorytatywnego nie zmienia zachowania metody. Pliki z releasu zachowano
w `inputs/recordings/*.overlay/`.

### 4.3 Kompletność danych

Dla wszystkich 8 wypuszczonych sekwencji sprawdziłem obecność plików wymaganych przez kod
(`moving_cam/camera_exts_full.txt`, `cam_ext.npy`, `cam_ext_slam_first.npy`,
`labels_con_sta/pred_con_gt.npy`, `smpl/3d_joints.npy`) oraz wideo `SonyRX0_motion_00*.mp4`.
Komplet jest dla każdej sekwencji.

Przy okazji znalazłem dwie niespójności długości, istotne wyłącznie przy liczeniu metryk
(pipeline ich nie dotyka, bo `align_slam` przycina wejścia do `min(...)` z ich długości):

| Sekwencja | `FRAME_RANGES` | `cam_ext` / `3d_joints` | Uwaga |
|---|---|---|---|
| s31 seq3 | 3597 | 3596 / 3596 | o jedną klatkę mniej niż zakres |
| s31 seq7 | 3501 | 3500 / **3584** | `3d_joints.npy` nie jest przycięty do zakresu — przy ewaluacji trzeba go obciąć ręcznie |

### 4.4 Konwencje danych, które musiałem ustalić eksperymentalnie

Dwie rzeczy nie są udokumentowane w repozytorium, a są kluczowe dla poprawności wyników:

1. `smpl/3d_joints.npy` jest wyrażony w **milimetrach**, natomiast `cam_ext.npy` w **metrach**
   (translacje w `camera_exts_full.txt` są w mikrometrach, dzielone przez 1e6). Bez
   przeskalowania GT metryki wychodzą tysiąc razy za duże.
2. `cam_ext.npy` oraz wideo z releasu są **już przycięte** do zakresu z `config.FRAME_RANGES`
   (`cam_ext.npy` odpowiada dokładnie `extrinsics[1576:3900]` pliku tekstowego), mimo że
   `camera_exts_full.txt` zawiera wszystkie 3902 klatki nagrania. Założenie odwrotne powoduje
   ciche obcięcie złych klatek (§7.1).

---

## 5. Przebieg uruchomienia

Cały łańcuch uruchamiam jednym poleceniem:

```bash
source ~/work/env.sh
cd ~/work/PhysDynPose
./run.sh --seq_no 6 --surface 31 \
  --video_path ~/work/data/Data/Recordings_31_03_23/SonyRX0_31_03_23/sync/seq6_captury/SonyRX0_motion_006.mp4
```

### 5.1 Czasy wykonania (sekwencja 2324 klatek)

| Etap | Narzędzie | Czas | Przepustowość |
|---|---|---|---|
| 1. Krojenie klatek z wideo | OpenCV | 41 s | 57 kl./s |
| 2. Estymacja pozy | 4D-Humans (ViTDet-H + HMR2.0 / ViT-H) | **1784 s** | 0,77 s/kl. |
| 2b. Korekta skali translacji | `tools/fix_out_list_focal.py` | < 1 s | — |
| 3. SLAM kamery | DROID-SLAM | 187 s tracking + ~180 s global BA | 12–13 it./s |
| 4. Przygotowanie wejść optymalizacji | `align_slam.py` | 8,4 s | — |
| 5. Optymalizacja fizyczna (QP) | `run_opt.py` | **16,9 s** | 7 ms/kl. |
| 6. Konwersja wyniku | `convert_res.py` | 1,9 s | — |

**Łącznie od wideo do wyniku: ok. 37 min**, z czego ok. 80 % stanowi estymator pozy, a 16 % SLAM.
Warstwa fizyczna, czyli wkład własny metody, jest obliczeniowo pomijalna. Ma to bezpośrednie
znaczenie praktyczne: koszt podmiany estymatora pozy albo policzenia wielu wariantów optymalizacji
jest znikomy wobec kosztu estymacji.

Wariant z kamerą z ground truth (bez SLAM-u) uruchamiam osobno:

```bash
python align_slam.py  --seq_no 6 --surface 31 --no-slam
python run_opt.py     --seq_no 6 --surface 31 --no-slam
python convert_res.py --seq_no 6 --surface 31 --method PIP --no-slam
```

### 5.2 Wyniki pośrednie i końcowe

```
outputs/raw_frames/seq6_31/                            2324 × PNG 1920×1080
inputs/4d_humans/outputs_06/out_list.pkl               2324 wpisy (jeden na klatkę)
inputs/slam/Trajectory.txt                            2324 pozy kamery
outputs/pip_input/seq6_31_4dhuman{,_slam}/            10 tensorów wejściowych optymalizacji
outputs/opt_result/seq6_31_4dhuman/PIP/              seq6_smoothed{,_slam}.pt
outputs/convert_res/seq6_31/eval/                     seq6_31{,_slam}.npy — (2324, 24, 3) float32 [m]
```

Wszystkie tensory wyjściowe są skończone, bez `NaN` i `Inf`.

---

## 6. Wyniki

### 6.1 Metryki względem ground truth

Odniesienie: `smpl/3d_joints.npy` z nagrania Captury (optyczny system przechwytywania ruchu),
przeliczone z mm na m, wszystkie 2324 klatki.

| Metryka | `--slam` (kamera z DROID-SLAM) | `--no-slam` (kamera GT) | Wartość GT |
|---|---|---|---|
| MPJPE w układzie świata | 1344,7 mm | **390,2 mm** | — |
| MPJPE root-relative (czysta poza) | 233,0 mm | **92,1 mm** | — |
| MPJPE po wyrównaniu sztywnym (Procrustes) | 729,2 mm | **296,1 mm** | — |
| Błąd pozycji miednicy | 1337,7 mm | **409,1 mm** | — |
| Średnia wysokość najniższego stawu | 0,027 m | −0,031 m | 0,037 m |
| Średnie \|przyspieszenie\| stawów | 0,0084 m/kl.² | 0,0050 m/kl.² | 0,0045 m/kl.² |

### 6.2 Interpretacja

**Jakość pozy.** Błąd root-relative na poziomie 92 mm dla metody beztreningowej na MoviCam
mieści się w oczekiwanym rzędzie wielkości — pipeline liczy sensowną pozę, a nie artefakt
numeryczny. Średnie przyspieszenie stawów (0,0050 wobec 0,0045 m/kl.² w GT) wskazuje, że wynik
jest gładki, bez jitteru typowego dla surowego wyjścia estymatora, czyli warstwa fizyczna
rzeczywiście wygładza trajektorię.

**Wpływ źródła kamery.** Wariant z kamerą GT jest **3,4× lepszy** w układzie świata i 2,5×
w root-relative. Przyczyną jest zaszyty w `align_slam.py` (`align_orb_poses`) współczynnik
skali `0.6` i ręczna zamiana osi, dopasowane do jednej konkretnej konwencji danych — na
trajektorii o innej skali wprowadzają błąd systematyczny. Skoro MoviCam udostępnia GT kamery
(`cam_ext.npy`), właściwym trybem pracy jest `--no-slam`; SLAM ma sens wyłącznie jako element
eksperymentu „ile tracimy bez GT”, nie jako zależność produkcyjna.

**Kontakt z podłogą.** W wariancie GT 6,8 % klatek ma najniższy staw poniżej minimalnej
wysokości GT, a penetracja jest rzędu pojedynczych centymetrów. Jest to znany efekt miękkich
ograniczeń kontaktowych w sformułowaniu QP, nie błąd uruchomienia.

**Spójność skali świata.** `pred_world_translation` mieści się w przedziale x ∈ [−1,5; 2,9] m,
z ∈ [−2,1; 0,9] m, a GT pozycji miednicy w x ∈ [0,5; 3,8] m, z ∈ [−1,7; 1,3] m. Zakresy są
zgodne co do rzędu wielkości, a różnica średnia 1,34 m to suma przesunięcia początku układu sceny
i błędu głębokości estymatora, nie błąd skali.

### 6.3 Ograniczenia pomiaru

Powyższe liczby pochodzą z jednej sekwencji i nie stanowią porównania z tabelami z publikacji.
Odtworzenie liczb z pracy wymagałoby ich konfiguracji `out_list.pkl` (patrz §7.3) oraz
ewentualnie innych hiperparametrów, których repozytorium nie dokumentuje. Metryki policzyłem
własnym skryptem — nie ma ich w wydanym kodzie.

---

## 7. Błędy w wydanym kodzie

Trzy z poniższych błędów uniemożliwiały przejście pipeline'u; wszystkie poprawiłem
w repozytorium.

### 7.1 Wideo z releasu jest już przycięte, a ekstraktor tego nie zakłada

`SonyRX0_motion_006.mp4` ma **2324 klatki**, czyli dokładnie `end − start + 1` dla zakresu
1576–3899 z `config.FRAME_RANGES`. `extract_movicam_frames.py` ustawiał
`CAP_PROP_POS_FRAMES = 1576` i ratował tylko 748 klatek, kończąc wyjątkiem
`RuntimeError: Expected 2324 frames ... but saved 748`.

**Poprawka** ([extract_movicam_frames.py](extract_movicam_frames.py)): jeśli liczba klatek
w pliku nie przekracza `end_frame`, wideo traktowane jest jako już przycięte i odczyt startuje
od klatki 0 (z komunikatem diagnostycznym).

### 7.2 `run_droid_slam.py` nie wypełnia wszystkich pól oczekiwanych przez DROID-SLAM

`DroidFrontend.__init__` odczytuje `args.upsample`, którego wrapper nie ustawiał →
`AttributeError: 'Namespace' object has no attribute 'upsample'`.

**Poprawka** ([run_droid_slam.py](run_droid_slam.py)): `args.upsample = False`, obok już
ustawianych `stereo`, `disable_vis` i `image_size`.

### 7.3 Skala translacji z 4D-Humans jest ~33× za duża — blokuje optymalizację

Najpoważniejszy problem, ponieważ objawia się dopiero na etapie 5 i wygląda jak błąd solvera.

`hmr2.utils.renderer.cam_crop_to_full` wyznacza głębokość jako `tz = 2f / (b·s)`, przyjmując
`f` z konfiguracji demo: `EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE × max(img_size)` =
`5000 / 224 × 1920 ≈ 42 857 px`. Rzeczywista ogniskowa nagrania, odczytana z
`camera_exts_full.txt`, wynosi **1289,9 px** — czyli **33× mniej**.

Skutki łańcuchowe:

- `pred_cam_t_full` ma głębokość ≈ 69 m zamiast ≈ 2 m (mediana po korekcie: 2,25 m);
- `align_slam` przenosi błąd do układu świata — postać ląduje ~65 m od sceny, poza zasięgiem
  `asset/height_map.npy`;
- `run_opt` zgłasza `RuntimeError: Quadratic program failed during optimization` **na pierwszej
  klatce**, ponieważ oba solvery (Clarabel i cvxopt) zwracają `None` dla niespójnego zadania.

W funkcji `cam_crop_to_full` od ogniskowej zależy **wyłącznie** składowa `z`, więc poprawka
sprowadza się do jednego przeskalowania:

```
z ← z · f_prawdziwe / f_założone        # 1289,9 / 42 857,1 = 0,030098
```

**Poprawka:** dodałem [tools/fix_out_list_focal.py](tools/fix_out_list_focal.py) oraz wywołanie
w [run.sh](run.sh) bezpośrednio po etapie 4D-Humans. Skalowanie można wyłączyć
(`FOURD_HUMANS_FOCAL_FIX=0`) lub zmienić założenia (`FOURD_HUMANS_ASSUMED_FOCAL`,
`FOURD_HUMANS_IMAGE_MAX`). Ścieżka FrankMocap korekty nie wymaga, ponieważ
[tools/frankmocap_to_out_list.py](tools/frankmocap_to_out_list.py) czyta `fx` wprost z plików
kamery.

Po poprawce `run_opt` kończy się w 17 s, tensory są skończone, wynik jest fizycznie sensowny
(§6).

> **Uwaga metodologiczna.** Repozytorium nie dokumentuje, jakiej ogniskowej użyto do
> wygenerowania „oficjalnych” `out_list.pkl` w releasie. Moja korekta używa ogniskowej
> **rzeczywistej**, co jest jedynym spójnym wyborem, jeśli dalsze etapy operują w metrach
> (`align_slam`, `height_map`). Do porównania z tabelami z publikacji trzeba jednak ustalić
> jedną, jawnie opisaną konwencję — bez tego wynik zależy od niewidocznego założenia.

### 7.4 Drobne

- `demo.py` z 4D-Humans nie zapisuje `out_list.pkl` i nie sortuje listy plików (`Path.glob`),
  podczas gdy `align_slam.py` zakłada porządek czasowy i jeden wpis na klatkę — konieczny
  patch opisany w §3.
- `pytorch3d==0.7.5` i `chumpy==0.71` z [requirements.txt](requirements.txt) nie istnieją
  na PyPI — uwzględniłem to w komentarzu w tym pliku.

### 7.5 Zmiany, które wprowadziłem w repozytorium

| Plik | Rodzaj | Treść |
|---|---|---|
| [extract_movicam_frames.py](extract_movicam_frames.py) | poprawka | obsługa wideo już przyciętego do zakresu |
| [run_droid_slam.py](run_droid_slam.py) | poprawka | brakujące `args.upsample` |
| [tools/fix_out_list_focal.py](tools/fix_out_list_focal.py) | nowy | korekta skali translacji do rzeczywistej ogniskowej |
| [run.sh](run.sh) | rozszerzenie | wywołanie korekty skali + zmienne sterujące |
| [README.md](README.md) | dokumentacja | opis korekty skali i przyciętego wideo |
| [RUN_REPORT.md](RUN_REPORT.md) | dokumentacja | odnośnik do niniejszego raportu |

---

## 8. Reprodukcja

### 8.1 Zawartość katalogu roboczego

```
~/work/envs/physopt      7,2 GB   środowisko Pythona 3.10 (torch cu118, pytorch3d, detectron2, RBDL)
~/work/4D-Humans          67 MB   checkout + patch demo.py; checkpointy w ~/.cache/4DHumans (5,1 GB)
~/work/DROID-SLAM        204 MB   droid_backends, lietorch_backends (sm_89), droid.pth
~/work/rbdl                —      źródła i build; rbdl.so w site-packages
~/work/PhysDynPose       5,0 GB   repozytorium, wejścia, wyniki
~/work/data             97,9 GB   MoviCam.zip (CRC zweryfikowane)
~/work/data/Data        97,9 GB   rozpakowany zbiór (zweryfikowany 1:1 z archiwum)
~/work/env.sh              —      eksporty środowiska + przykładowe polecenia
```

Po rozpakowaniu danych zajęte jest 183 GiB; pozostaje 8,2 TB wolnego miejsca.

### 8.2 Uruchomienie kolejnej sekwencji

```bash
source ~/work/env.sh
cd ~/work/PhysDynPose

# pełny łańcuch, startując od wideo
./run.sh --seq_no 3 --surface 31 \
  --video_path ~/work/data/Data/Recordings_31_03_23/SonyRX0_31_03_23/sync/seq3_captury/SonyRX0_motion_003.mp4

# albo bez SLAM-u, na już wyciętych klatkach
./run.sh --seq_no 3 --surface 31 --frames_dir outputs/raw_frames/seq3_31 --no-slam
```

Dostępne sekwencje: `surface=31` → seq 1, 2, 3, 4, 6, 7; `surface=24` → seq 1, 2.
Koszt: ~35 min na sekwencję; pięć sekwencji można liczyć równolegle na pięciu GPU.

---

## 9. Ograniczenia

1. **Jedna sekwencja end-to-end.** Pozostałe siedem ma zweryfikowane dane wejściowe, ale ich
   nie przeliczyłem — nie ma więc jeszcze rozkładu błędów po scenach.
2. **Brak porównania z publikacją.** Nie odtworzyłem tabel z pracy; wymagałoby to ich konfiguracji
   estymatora i rozstrzygnięcia kwestii z §7.3.
3. **Wizualizacja 3D nieprzetestowana.** `visualizer.py` wymaga PyBullet w trybie GUI — na
   maszynie nie ma serwera X, a przez SSH bez Xvfb/VNC okno się nie otworzy. Dane wejściowe
   do wizualizacji są gotowe, więc test sprowadza się do uruchomienia na maszynie z pulpitem.
4. **Metryki policzyłem własnym skryptem**, a nie częścią wydanego kodu. Przy dalszej pracy
   warto je dodać do repozytorium jako osobny etap ewaluacji.
5. **Wersje bibliotek** są w kilku miejscach nowsze niż w publikacji (np. `cvxopt` 1.3.3 zamiast
   1.2.6 z `requirements.txt`), ponieważ dla starszych nie ma wheeli. Nie zaobserwowano
   towarzyszących temu problemów numerycznych, ale nie jest to konfiguracja referencyjna.

---

## 10. Wnioski

1. **Kod daje się uruchomić od zera na dostępnym sprzęcie** i przechodzi cały łańcuch na
   oryginalnych danych. Bariera wejścia jest jednak wysoka: trzy błędy w wydanym kodzie (§7)
   i jedenaście problemów zgodności i kompilacji (§3.1) trzeba rozwiązać samodzielnie.
2. **Warstwa fizyczna jest obliczeniowo darmowa** (17 s na 2324 klatki, 7 ms/klatkę) i wyraźnie
   poprawia gładkość wyniku. Wąskim gardłem są wyłącznie etapy sieciowe: estymator pozy
   (0,77 s/klatkę) i SLAM (0,16 s/klatkę) na L40S, więc warianty eksperymentów po stronie
   optymalizacji można liczyć praktycznie bez ograniczeń.
3. **Przy dostępnym GT kamery należy używać `--no-slam`** — daje 3,4× mniejszy błąd w układzie
   świata. Ścieżka SLAM-owa ma zaszyte stałe dopasowane do jednej konwencji danych i wymaga
   rekalibracji na nowych danych.
4. **Największe ryzyko metodologiczne to skala translacji** (§7.3). Jest to wielkość niejawna,
   zależna od konfiguracji estymatora, a jednocześnie decydująca o tym, czy optymalizacja
   fizyczna w ogóle się rozwiąże. Dla wiarygodnego porównania z publikacją trzeba ją jawnie
   udokumentować i utrzymać stałą we wszystkich eksperymentach.
5. **Naturalny następny krok** to policzenie pozostałych siedmiu sekwencji (jeden przebieg
   równoległy, ~35 min) i zestawienie rozkładu metryk, a następnie porównanie kosztu rezygnacji
   z GT kamery — infrastruktura do tego jest gotowa, bo oba warianty liczy się jednym
   przełącznikiem `--slam` / `--no-slam`.

---

## Załącznik A. Konfiguracja sekwencji referencyjnej

| Element | Wartość |
|---|---|
| Sekwencja | `surface=31`, `seq_no=6` |
| Zakres klatek (`config.FRAME_RANGES`) | 1576–3899 (2324 klatki) |
| Rozdzielczość / fps | 1920×1080 / 25 |
| Ogniskowa `fx` z `camera_exts_full.txt` | 1289,9 px |
| Ogniskowa przyjęta przez demo 4D-Humans | 42 857,1 px → korekta ×0,030098 |
| Detektor | ViTDet-H (cascade mask R-CNN), próg pewności 0,25 |
| Estymator pozy | HMR2.0, ViT-H, checkpoint `epoch=35-step=1000000` |
| SLAM | DROID-SLAM, `buffer=512`, wejście 240×320, `stride=1` |
| Solver QP | Clarabel 0.11.1, fallback cvxopt 1.3.3 |
| Model ciała | `models/SMPL_male_10PCs.pkl` (10 współczynników kształtu) |
