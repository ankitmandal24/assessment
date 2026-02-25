# Feature Measurements — Complete Column-by-Column Analysis

> Source code: `automorph/measure/measure.py` and `automorph/measure/get_vessel_coords.py`  
> Output file: `M3/feature_measurements.csv`

This document explains **every column** saved in `feature_measurements.csv` — what it means in simple language, where in the code it comes from, the **code formula**, and the **real-life mathematical formula**.

---

## How the Pipeline Works (Brief Overview)

1. **Load the image** (fundus photograph of the retina, 912×912 px).
2. **Load segmentation masks** — binary vessels, artery/vein map, and optic disc/cup.
3. **Measure optic disc & cup** → disc dimensions, cup dimensions, CDR, laterality.
4. **Detect individual vessel paths** — skeletonize the vessel map, remove branch points, split into separate segments, and order them (done in `get_vessel_coords.py`).
5. **Define measurement zones** — whole image, Zone B (ring from 2× to 3× OD radius), Zone C (ring from 2× to 5× OD radius).
6. **Measure vessel features** — density, fractal dimension, calibre, tortuosity for each zone × vessel type (binary/artery/vein).
7. **Compute CRAE, CRVE, AVR** — retinal artery and vein central equivalents using the Knudtson formula.
8. **Save everything** to `feature_measurements.csv`.

---

## Part A — Optic Disc & Cup Columns

> Code function: `get_disc_metrics()` (line 353)

---

### `disc_height`

**What it means:** The vertical size (in pixels) of the optic disc.

**How it's calculated:** Find all pixels where the disc mask is > 0. The height is the difference between the topmost and bottommost pixel row.

**Code (line 364):**
```python
disc_height = np.max(disc_index_y) - np.min(disc_index_y)
```

**Math:**
$$H_{disc} = y_{max} - y_{min}$$

---

### `disc_width`

**What it means:** The horizontal size (in pixels) of the optic disc.

**Code (line 365):**
```python
disc_width = np.max(disc_index_x) - np.min(disc_index_x)
```

**Math:**
$$W_{disc} = x_{max} - x_{min}$$

---

### `cup_height`

**What it means:** The vertical size (in pixels) of the optic cup (the pale region inside the disc).

**Code (line 375):**
```python
cup_height = np.max(cup_index_y) - np.min(cup_index_y)
```

**Math:**
$$H_{cup} = y_{max} - y_{min}$$

---

### `cup_width`

**What it means:** The horizontal size (in pixels) of the optic cup.

**Code (line 376):**
```python
cup_width = np.max(cup_index_x) - np.min(cup_index_x)
```

**Math:**
$$W_{cup} = x_{max} - x_{min}$$

---

### `CDR_vertical` (Vertical Cup-to-Disc Ratio)

**What it means:** How much of the disc's height is occupied by the cup. A higher value can indicate glaucoma risk.

**Code (line 383):**
```python
CDR_vertical = cup_height / disc_height
```

**Math:**
$$CDR_v = \frac{H_{cup}}{H_{disc}}$$

---

### `CDR_horizontal` (Horizontal Cup-to-Disc Ratio)

**What it means:** How much of the disc's width is occupied by the cup.

**Code (line 384):**
```python
CDR_horizontal = cup_width / disc_width
```

**Math:**
$$CDR_h = \frac{W_{cup}}{W_{disc}}$$

---

### `macular_centred`

**What it means:** Is this photograph centered on the **macula** (True) or on the **optic disc** (False)? The code checks whether the optic disc centroid is within 10% of the image center — if it is, the photo is disc-centred.

**Code (lines 409–413):**
```python
horizontal_distance = abs(mean(disc_y) - img_size/2)
vertical_distance   = abs(mean(disc_x) - img_size/2)
distance_ = sqrt(horizontal_distance**2 + vertical_distance**2)
macular_centred = (distance_ / img_size) >= 0.1   # True → macula-centred
```

**Math:**
$$d = \sqrt{\left(\bar{y}_{disc} - \frac{S}{2}\right)^2 + \left(\bar{x}_{disc} - \frac{S}{2}\right)^2}$$

$$\text{macular\_centred} = \begin{cases} \text{False (disc-centred)} & \text{if } \frac{d}{S} < 0.1 \\ \text{True (macula-centred)} & \text{otherwise} \end{cases}$$

---

### `laterality`

**What it means:** Which eye — Left or Right? The code decides by comparing vessel density on each side of the optic disc. More vessels face toward the macula (temporal side).

**Code (lines 417–419):**
```python
if vessel_mask[:, :cup_centre_x].sum() > vessel_mask[:, cup_centre_x:].sum():
    laterality = 'Right'
else:
    laterality = 'Left'
```

**Math:**
$$\text{laterality} = \begin{cases} \text{Right} & \text{if } \displaystyle\sum_{x < x_c} V(x,y) > \sum_{x \ge x_c} V(x,y) \\ \text{Left} & \text{otherwise} \end{cases}$$

---

## Part B — Global Vessel Columns (whole image)

> Code function: `global_metrics()` (line 56) and `vessel_metrics()` (line 207)

These are computed once per vessel type (binary / artery / vein) on the **whole** image.

---

### `vessel_density`

**What it means:** The fraction of the image that is covered by blood vessels. Higher = denser network.

**Code (lines 221–223):**
```python
vessel_total_count = np.sum(vessels == 1)
pixel_total_count  = vessels.shape[0] * vessels.shape[1]
vessel_density     = vessel_total_count / pixel_total_count
```

**Math:**
$$VD = \frac{\text{number of vessel pixels}}{\text{total pixels in image}}$$

---

### `fractal_dimension`

**What it means:** A number describing how **complex and space-filling** the vessel branching pattern is. Healthy retinas typically have a fractal dimension around 1.4–1.7. It uses the **Box-Counting method**.

**Technique:** Overlay the vessel image with grids of decreasing box sizes. For each box size $k$, count how many boxes touch at least one vessel pixel (but are not fully filled). Plot $\log(\text{count})$ vs $\log(\text{box size})$ and fit a straight line. The slope (negated) is the fractal dimension.

**Code (lines 33–52):**
```python
# Box counting function
def boxcount(Z, k):
    S = np.add.reduceat(np.add.reduceat(Z, np.arange(0,Z.shape[0],k), axis=0),
                        np.arange(0,Z.shape[1],k), axis=1)
    return len(np.where((S > 0) & (S < k*k))[0])

# Fractal dimension
sizes = 2**np.arange(n, 1, -1)         # e.g. [512, 256, 128, ..., 4]
counts = [boxcount(Z, size) for size in sizes]
coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
fractal_dimension = -coeffs[0]          # negative slope
```

**Math:**
$$D_f = -\frac{d(\log N(k))}{d(\log k)} \approx -\text{slope of } \log N(k) \text{ vs } \log k$$

where $N(k)$ = number of boxes of side $k$ that intersect the vessel.

---

### `average_global_calibre`

**What it means:** The average **thickness** (width) of all vessels, estimated globally. The code divides the total vessel area by the length of the skeleton (thinned 1-pixel-wide centerline). Think of it like: total area ÷ total length ≈ average width.

**Code (lines 58–63):**
```python
skeleton = morphology.skeletonize(vessels)        # 1-pixel centreline
average_width = np.sum(vessel_) / np.sum(skeleton) # area / length
```

**Math:**
$$\bar{W}_{global} = \frac{\sum \text{vessel pixels}}{\sum \text{skeleton pixels}}$$

---

## Part C — Zonal Vessel Columns (per zone × per vessel type)

> Code function: `vessel_metrics()` (line 207)

These measurements happen in **three zones**:
- **whole** — entire image  
- **Zone B** — ring from 2× to 3× optic disc radius around the disc center  
- **Zone C** — ring from 2× to 5× optic disc radius around the disc center  

And for **three vessel types**: `binary` (all vessels), `artery`, `vein`.

Column names follow the pattern: `{metric}_{vessel_type}_{zone}`, e.g. `tortuosity_distance_artery_B`.

---

### `average_local_calibre`

**What it means:** The average vessel **width** measured precisely at each point along each individual vessel segment, then averaged. Unlike global calibre (area ÷ length), this approach traces each vessel path, finds the two edges at every point, and measures the perpendicular distance between them.

**Technique (code lines 141–202):**
1. **Distance Transform** (`cv2.distanceTransform`) — for each vessel pixel, how far is it from the nearest edge? This gives the half-width at each point.
2. **Gradient + Normal Direction** — compute the tangent direction along the skeleton path using `np.gradient`, then rotate by 90° to get the normal (perpendicular) direction.
3. **Project edge points** — move outward from the skeleton along the normal direction by the distance-transform value to find the two edges.
4. **Canny Edge Detection** — refine the exact edge positions using `cv2.Canny`.
5. **KDTree matching** — snap projected edge points to actual edge pixels using nearest-neighbour search.
6. **Width** = Euclidean distance between opposing edge points: `np.linalg.norm(e1 - e2)`.
7. Average across all points on all vessels.

**Code (line 199–200):**
```python
widths = [np.linalg.norm(e1 - e2, axis=1) for e1, e2 in zip(edges1, edges2)]
avg_width = [np.mean(w) for w in widths]
```

**Math:**
$$\bar{W}_{local} = \frac{1}{M} \sum_{j=1}^{M} \frac{1}{N_j} \sum_{i=1}^{N_j} \|\mathbf{e}^+_{j,i} - \mathbf{e}^-_{j,i}\|_2$$

where $M$ = number of vessels, $N_j$ = number of points on vessel $j$, $\mathbf{e}^+$ and $\mathbf{e}^-$ = opposing edge coordinates.

---

### `tortuosity_distance` (Curve-to-Chord Ratio)

**What it means:** How **wiggly** are the vessels? A perfectly straight vessel has a value of 1.0. The more curved/twisted it is, the higher the value. The code sums up the actual path length and divides by the straight-line chord from start to end.

**Code (lines 258–261, 272–273):**
```python
v_length = _curve_length(vessel[0], vessel[1])   # actual vessel length
c_length = _chord_length(vessel[0], vessel[1])   # straight-line start→end
tcc += v_length / c_length                        # per vessel
tcc = tcc / vessel_count                           # average
```

**Sub-formulas used:**
```python
# Curve length (sum of segment-by-segment Euclidean distances)
def _curve_length(x, y):
    return np.sum(((x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2)**0.5)

# Chord length (Euclidean distance from first to last point)
def _chord_length(x, y):
    return sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
```

**Math:**

$$L_{curve} = \sum_{i=1}^{N-1} \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}$$

$$L_{chord} = \sqrt{(x_N - x_1)^2 + (y_N - y_1)^2}$$

$$T_{dist} = \frac{1}{M}\sum_{j=1}^{M} \frac{L_{curve,j}}{L_{chord,j}}$$

---

### `tortuosity_density`

**What it means:** A more detailed tortuosity metric that accounts for **local twists**. Instead of looking at the whole vessel at once, it first finds **inflection points** (where the vessel changes direction of curvature), splits the vessel into segments at those points, and sums up the tortuosity of each sub-segment. This captures vessels that twist back and forth frequently.

**Technique:** Based on the method from [Grisan et al., IEEE 2003](https://ieeexplore.ieee.org/document/1279902).

**Step 1 — Find inflection points (line 97–103):**
```python
cf = np.convolve(y, [1, -1])   # first differences of y-coordinates
# An inflection point occurs where the sign of cf changes
for i in range(2, len(x)):
    if np.sign(cf[i]) != np.sign(cf[i-1]):
        inflection_points.append(i-1)
```

**Step 2 — Sum segment tortuosities (lines 116–126):**
```python
for in_point in inflection_points:
    seg_curve = _curve_length(segment_x, segment_y)
    seg_chord = _chord_length(segment_x, segment_y)
    if seg_chord:
        sum_segments += seg_curve / seg_chord - 1

tortuosity_density = (n-1)/n + (1/curve_length) * sum_segments
```

**Math:**

$$T_{dens} = \frac{n-1}{n} + \frac{1}{L_{curve}} \sum_{i=1}^{n} \left(\frac{L_{curve,i}}{L_{chord,i}} - 1\right)$$

where $n$ = number of inflection-point segments.

---

## Part D — Large Vessel Equivalent Columns

> Code: `Knudtson_cal()` (line 75) and inside `vessel_metrics()` (lines 287–346)

These are computed **only for artery and vein maps** in **Zones B and C** (not whole image, not binary).

---

### `CRAE_Knudtson_artery_B` / `CRAE_Knudtson_artery_C`
**(Central Retinal Arteriolar Equivalent)**

**What it means:** An estimate of the width of the main central retinal artery *before* it branches. This is a key **cardiovascular risk biomarker**. Narrower arteries (lower CRAE) may indicate hypertension.

**Technique (Knudtson revised formula):**
1. Measure the widths of all individual arteries in the zone.
2. Take the **6 largest** widths and sort them: $w_1 \le w_2 \le \dots \le w_6$.
3. **Round 1** — Pair them and combine:
   - Pair $(w_1, w_6)$ → combined width
   - Pair $(w_2, w_5)$ → combined width
   - Pair $(w_3, w_4)$ → combined width
4. **Round 2** — Sort the 3 results, pair again: combine (smallest, largest), keep middle.
5. **Round 3** — Combine the last 2 values → final CRAE.

**Pairing Code (line 75–78):**
```python
def Knudtson_cal(w1, w2):
    w_artery = 0.88 * np.sqrt(w1**2 + w2**2)
    w_vein   = 0.95 * np.sqrt(w1**2 + w2**2)
    return w_artery, w_vein
```

**Math (Knudtson artery formula):**
$$W_{a} = 0.88 \cdot \sqrt{w_1^2 + w_2^2}$$

Reference: Knudtson MD, Lee KE, Hubbard LD, et al. *Revised formulas for summarizing retinal vessel diameters.* Current Eye Research, 2003.

---

### `CRVE_Knudtson_vein_B` / `CRVE_Knudtson_vein_C`
**(Central Retinal Venular Equivalent)**

**What it means:** An estimate of the width of the main central retinal vein. Wider veins (higher CRVE) may indicate inflammation or diabetes risk.

**Technique:** Same iterative pairing procedure as CRAE, but using the vein formula with coefficient 0.95 instead of 0.88.

**Math (Knudtson vein formula):**
$$W_{v} = 0.95 \cdot \sqrt{w_1^2 + w_2^2}$$

---

### `AVR_B` / `AVR_C`
**(Arteriovenular Ratio)**

**What it means:** The ratio of artery width to vein width. A healthy eye typically has AVR between 0.6–0.8. Lower AVR suggests arteriolar narrowing (hypertension risk).

**Code (lines 442–448):**
```python
AVR_B = CRAE_Knudtson_artery_B / CRVE_Knudtson_vein_B
AVR_C = CRAE_Knudtson_artery_C / CRVE_Knudtson_vein_C
```

**Math:**
$$AVR = \frac{CRAE}{CRVE}$$

---

## Part E — Hubbard Formula (Alternative, defined but not used by default)

The code also defines a **Hubbard–Parr formula** (line 68–71), which is an older approach for computing retinal vessel equivalents. It is defined but **not called** in the main pipeline — the newer Knudtson formula is used instead.

```python
def Hubbard_cal(w1, w2):
    w_artery = sqrt(0.87*w1² + 1.01*w2² - 0.22*w1*w2 - 10.76)
    w_vein   = sqrt(0.72*w1² + 0.91*w2² + 450.05)
    return w_artery, w_vein
```

**Math (Hubbard–Parr):**
$$W_a = \sqrt{0.87 w_1^2 + 1.01 w_2^2 - 0.22 w_1 w_2 - 10.76}$$
$$W_v = \sqrt{0.72 w_1^2 + 0.91 w_2^2 + 450.05}$$

---

## Part F — Vessel Skeleton Extraction (Pre-processing step)

> Code: `get_vessel_coords.py` → `generate_vessel_skeleton()`

Before any vessel metric is computed, the code extracts individual vessel path coordinates to pass to the measurement functions.

| Step | What happens | Technique |
|---|---|---|
| 1 | Remove optic disc from vessel map | Masking (`vessels[od_mask > 0] = 0`) |
| 2 | Close small gaps in vessels | Morphological closing (`cv2.MORPH_CLOSE` with disk kernel r=3) |
| 3 | Thin vessels to 1-pixel skeleton | `skimage.morphology.skeletonize` |
| 4 | Remove branch/junction points | Convolve skeleton with 3×3 kernel of ones; any pixel with ≥3 neighbours is a junction → remove it |
| 5 | Remove short fragments | Discard any connected component shorter than 10 pixels |
| 6 | Order coordinates | Find endpoints (pixels with exactly 1 neighbour), sort by distance from optic disc center, then DFS traversal to produce ordered start→end paths |

---

## Part G — Zone Definitions

> Code: `utils.generate_zonal_masks()` (line 312 of `utils.py`)

| Zone | Inner boundary | Outer boundary |
|---|---|---|
| **whole** | none | entire image |
| **B** | 2 × OD radius | 3 × OD radius |
| **C** | 2 × OD radius | 5 × OD radius |

Both Zone B and C are **annular rings** (doughnut shapes) centered on the optic disc. Zone B is the narrow inner ring, Zone C is the wider ring extending further out.

---

## Quick Reference Table — All Columns

| Column | Category | Formula | Code Line |
|---|---|---|---|
| `disc_height` | Disc | $y_{max} - y_{min}$ | 364 |
| `disc_width` | Disc | $x_{max} - x_{min}$ | 365 |
| `cup_height` | Cup | $y_{max} - y_{min}$ | 375 |
| `cup_width` | Cup | $x_{max} - x_{min}$ | 376 |
| `CDR_vertical` | Cup/Disc | $H_{cup}/H_{disc}$ | 383 |
| `CDR_horizontal` | Cup/Disc | $W_{cup}/W_{disc}$ | 384 |
| `macular_centred` | Disc | Euclidean distance check < 10% | 409–413 |
| `laterality` | Disc | Vessel density comparison left vs right | 417–419 |
| `vessel_density` | Vessel | vessel pixels / total pixels | 221–223 |
| `fractal_dimension` | Vessel | Box-counting log-log slope | 33–52 |
| `average_global_calibre` | Vessel | vessel area / skeleton length | 62 |
| `average_local_calibre` | Vessel | Mean edge-to-edge distance per point | 199–200 |
| `tortuosity_distance` | Vessel | Mean(curve length / chord length) | 258–273 |
| `tortuosity_density` | Vessel | Inflection-point segmented tortuosity | 107–126 |
| `CRAE_Knudtson` | Large vessel | $0.88\sqrt{w_1^2+w_2^2}$ iteratively | 75–77, 310–332 |
| `CRVE_Knudtson` | Large vessel | $0.95\sqrt{w_1^2+w_2^2}$ iteratively | 75–78, 336–342 |
| `AVR` | Large vessel | CRAE / CRVE | 442–448 |
