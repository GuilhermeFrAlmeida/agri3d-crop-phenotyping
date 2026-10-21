## 1. First decision: what data are we actually going to use?

I recommend making **MaizeField3D the primary dataset**.

More specifically, our **model-development input** will be:

> **MaizeField3D → `FielGrwon_ZeaMays_RawPCD_100k.zip`**

It contains **1,045 field-grown maize plant point clouds**, each represented by 100k points, and the data were acquired with a **Faro Focus S350 terrestrial laser scanner**. The associated paper says the plants originated from a maize diversity panel at Iowa State University, were harvested around anthesis, and then scanned indoors after being collected from the field. ([PubMed Central (PMC)][1])

The important distinction is that these are **raw/unsegmented point clouds from the dataset**, not the original scanner acquisition files. They are `.ply` point clouds already extracted at the plant level and downsampled to 100k/50k/10k variants. ([Hugging Face][2])

That is actually a good compromise for a 4–6 week project.

### Why I'm choosing it

We get all of these in one dataset:

```text
1,045 plants
        │
        ├── raw/unsegmented point clouds
        │
        ├── 520 segmented/annotated plants
        │      ├── leaves
        │      └── stalk
        │
        ├── metadata
        │
        └── procedural plant models
```

The 520 annotated plants are especially useful because they give us actual targets for semantic/instance segmentation rather than forcing us to invent labels. The dataset also provides 100k, 50k and 10k versions. ([Baskar Group][3])

One caveat: the Hugging Face distribution currently lists the dataset under **CC BY-NC 4.0**, so we should treat this as a research/portfolio project and preserve attribution rather than presenting the dataset itself as commercially redistributable. ([Hugging Face][2])

---

# 2. But I want one important addition

I don't want our project to stop at:

```text
PLY → PointNet++ → segmentation
```

That would demonstrate 3D deep learning, but not enough **LiDAR engineering**.

So I'd design the project with **two data layers**.

### Layer A — training/phenotyping dataset

**MaizeField3D**

```text
PLY
 ↓
point-cloud processing
 ↓
semantic/instance segmentation
 ↓
plant reconstruction
 ↓
phenotypic metrics
```

### Layer B — sensor-native validation

Use a small portion of **Pullally Dataset** later as an integration/robustness test.

That dataset contains **raw LiDAR + GNSS-RTK recordings** collected by a quadruped robot in an actual Chilean orchard, using a Velodyne VLP-16, with approximately 75 GB of ROS bag data. ([GitHub][4])

That gives us a second pipeline:

```text
ROS bag
 ↓
LiDAR packets
 ↓
point cloud extraction
 ↓
motion / pose information
 ↓
registration
 ↓
field-level 3D map
 ↓
our trained 3D perception components
```

We don't need to fully solve orchard segmentation there.

The purpose is to demonstrate:

> **“I understand both point-cloud ML and how LiDAR data actually comes out of a robotic sensing system.”**

That's much stronger professionally.

---

# 3. The project we should actually build

I'd define the project as:

# Agri3D

### End-to-End LiDAR-Based Crop Structure Analysis

The first release should answer:

> **Given a 3D scan of a maize plant, can we automatically identify its structural components and convert the 3D geometry into useful phenotypic measurements?**

The architecture:

```text
                   INPUT
                     │
            MaizeField3D PLY
                     │
                     ▼
          ┌────────────────────┐
          │ Point Cloud Loader │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Quality Processing │
          │ - outliers         │
          │ - normalization    │
          │ - voxelization     │
          │ - sampling         │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ 3D Representation  │
          │ XYZ + features     │
          └─────────┬──────────┘
                    │
                    ▼
       ┌───────────────────────────┐
       │      3D Deep Learning     │
       │                           │
       │ PointNet++ → baseline     │
       │ Point Transformer/KPConv  │
       │          → final          │
       └────────────┬──────────────┘
                    │
                    ▼
            Semantic labels
         ┌──────────┴──────────┐
         │                     │
       stalk                  leaf
         │                     │
         └──────────┬──────────┘
                    ▼
          Instance reconstruction
                    │
                    ▼
          Geometric measurements
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        height    leaf count   volume
                    │
                    ▼
             Phenotype report
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
         CSV      3D cloud    report
```

---

# 4. Phase 0 — define the engineering contract

Before implementing the model, I'd create a small document:

```text
docs/
└── project_specification.md
```

with five things.

### Input contract

```text
Input:
  .ply

Required:
  x, y, z

Optional:
  r, g, b
  intensity
  normals
```

For the first version, our model should use:

```text
XYZ
```

only.

That's deliberate.

We want to know whether the network can learn **3D structure itself**, rather than accidentally relying on color.

We can later introduce:

```text
XYZ
XYZ + normals
XYZ + RGB
```

as an ablation study.

### Output contract

Every prediction should ultimately become:

```json
{
  "plant_id": "000123",
  "point_count": 100000,
  "height_m": 2.31,
  "leaf_count": 12,
  "stalk_points": 18342,
  "leaf_points": 81658,
  "canopy_volume_m3": 0.41
}
```

This makes the project feel like a system rather than a notebook.

---

# 5. Phase 1 — understand the point clouds

Before touching deep learning, we'll build a classical processing pipeline.

### 5.1 Loader

Use:

```text
Open3D
numpy
plyfile
```

and implement:

```python
load_point_cloud()
validate_point_cloud()
inspect_attributes()
```

We'll establish:

```text
number of points
bounding box
XYZ range
density
missing values
duplicate points
```

For every cloud we should produce something like:

```text
points:       100,000
x range:      ...
y range:      ...
z range:      ...
height:       ...
density:      ...
```

### 5.2 Coordinate normalization

We don't want the network learning scanner/world coordinates.

Convert:

```text
P' = P - centroid
```

and optionally align the plant so the vertical axis is consistent.

We should investigate PCA, but not blindly use PCA as “vertical”; because plant geometry can make the largest eigenvector a poor estimate of gravity.

So we'll distinguish:

```text
translation normalization
rotation normalization
scale normalization
```

and test them independently.

### 5.3 Sampling

Because every cloud already has 100k/50k/10k variants, we have a very useful experiment:

```text
10k
50k
100k
```

and measure:

```text
accuracy
GPU memory
training time
inference time
```

This is excellent material for the final engineering report.

---

# 6. Phase 2 — build a non-DL baseline

This is important.

Before PointNet++, implement a geometric baseline.

Something like:

```text
point cloud
 ↓
normal estimation
 ↓
height-based partition
 ↓
geometric clustering
 ↓
stalk/leaf candidate separation
```

The exact algorithm can evolve after we inspect the data.

Potential tools:

```text
Open3D
scikit-learn
DBSCAN
RANSAC
PCA
KDTree
```

The purpose isn't necessarily to achieve the best score.

The purpose is to establish:

> **How much can conventional 3D geometry accomplish before deep learning?**

Then we have a meaningful comparison:

```text
Classical geometry
        VS
PointNet++
        VS
Point Transformer
```

---

# 7. Phase 3 — first neural network

Now we train **PointNet++**.

Input:

```text
N × 3
```

Output:

```text
N × C
```

where initially:

```text
C = 2

0 = leaf
1 = stalk
```

We deliberately keep the first problem simple.

### Metrics

We'll report:

```text
overall accuracy
mean IoU
per-class IoU
precision
recall
F1
```

But **mIoU will be the primary segmentation metric**.

We should also visualize predictions:

```text
Ground truth      Prediction
     ↓                ↓

    3D                 3D
colored              colored
```

This is critical for debugging 3D models.

---

# 8. Phase 4 — instance structure

This is where the project gets more interesting.

Semantic segmentation tells us:

```text
leaf
leaf
leaf
leaf
stalk
stalk
```

but we want:

```text
leaf #1
leaf #2
leaf #3
...
stalk
```

So after semantic segmentation we'll develop an instance extraction stage.

Potential approach:

```text
leaf points
   ↓
graph / neighborhood construction
   ↓
connected components
   ↓
geometric merging
   ↓
individual leaves
```

Eventually:

```text
Plant
 ├── stalk
 ├── leaf_01
 ├── leaf_02
 ├── leaf_03
 ├── ...
 └── leaf_N
```

This is where the project stops being a generic point-cloud segmentation project and becomes an agricultural phenotyping system.

---

# 9. Phase 5 — phenotype extraction

Now we convert geometry into actual measurements.

### Plant height

```text
height = max(z) - min(z)
```

after proper normalization.

### Leaf count

```text
N_leaves = number of extracted leaf instances
```

### Leaf length

Several possible approaches:

```text
PCA major axis
     or
skeleton length
     or
geodesic distance
```

We'll evaluate which makes sense given the actual point-cloud geometry.

### Canopy volume

We can compare:

```text
convex hull
alpha shape
voxel occupancy
Poisson mesh
```

This itself becomes a useful 3D geometry experiment.

For example:

```text
Point cloud
      │
      ├── convex hull
      ├── alpha shape
      └── voxel volume
```

and investigate how much those estimates differ.

---

# 10. Phase 6 — deep-learning comparison

Once the complete pipeline works, we'll replace the baseline network.

I'd test one architecture from the modern family:

### Option A

**Point Transformer**

### Option B

**KPConv**

We don't need five models.

One baseline + one stronger architecture is enough.

Then report:

```text
                PointNet++    Advanced model
------------------------------------------------
mIoU               ...             ...
Leaf IoU            ...             ...
Stalk IoU           ...             ...
Inference           ...             ...
GPU memory          ...             ...
Parameters          ...             ...
```

Not just “model X is better.”

We should explicitly analyze the **accuracy/compute tradeoff**.

---

# 11. Phase 7 — robustness experiments

This is where I think your existing ML background becomes useful.

We deliberately introduce perturbations:

```text
Point dropout
Gaussian coordinate noise
Different point densities
Rotation
Occlusion
```

For example:

```text
100k points
 ↓
50k
 ↓
10k
 ↓
5k
 ↓
random 30% removal
```

and observe:

```text
mIoU vs point density
```

This makes the project feel much closer to a real sensing problem.

---

# 12. Phase 8 — sensor-native LiDAR pipeline

Only after the phenotyping system works should we touch Pullally.

We take:

```text
ROS bag
```

and create:

```text
LiDAR frames
 +
GNSS
```

Then:

```text
ROS bag
   ↓
point cloud extraction
   ↓
timestamp synchronization
   ↓
coordinate transformation
   ↓
trajectory alignment
   ↓
registration
   ↓
global point cloud
```

The dataset uses a VLP-16 and RTK-GNSS, so this gives us experience with an entirely different class of LiDAR data than the static TLS plant scans used by MaizeField3D. ([GitHub][4])

We don't need to train a new model here.

The point is to prove that our pipeline can accept **sensor-native 3D data**, not only curated PLY files.

---

# 13. Phase 9 — industrial interface

At the end:

```text
input
  ↓
pipeline
  ↓
results
```

should be callable through:

```bash
python -m agri3d.pipeline \
    --input plant.ply \
    --model weights/model.pth \
    --output output/
```

and optionally:

```bash
POST /analyze
```

with:

```text
multipart/form-data
    point_cloud = plant.ply
```

returning:

```json
{
  "plant_height_m": 2.31,
  "leaf_count": 12,
  "canopy_volume_m3": 0.41,
  "processing_time_ms": 183
}
```

This part is especially worthwhile for you because it connects naturally with your existing Python/Flask/deployment experience.

---

# 14. Proposed repository

I'd start with this:

```text
agri3d/
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── inference.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   └── agri3d/
│       ├── io/
│       │   ├── ply.py
│       │   └── rosbag.py
│       │
│       ├── preprocessing/
│       │   ├── filtering.py
│       │   ├── normalization.py
│       │   ├── sampling.py
│       │   └── features.py
│       │
│       ├── geometry/
│       │   ├── ground.py
│       │   ├── clustering.py
│       │   ├── skeleton.py
│       │   └── volume.py
│       │
│       ├── models/
│       │   ├── pointnet2.py
│       │   └── point_transformer.py
│       │
│       ├── training/
│       │   ├── dataset.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── losses.py
│       │
│       ├── phenotyping/
│       │   ├── instances.py
│       │   ├── traits.py
│       │   └── report.py
│       │
│       ├── visualization/
│       │   ├── pointcloud.py
│       │   └── metrics.py
│       │
│       └── api/
│           └── app.py
│
├── tests/
├── notebooks/
├── scripts/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

# 15. Dataset split — one thing we should be careful about

This is important enough that I would make it an explicit Phase 0 task.

We **should not simply randomly split individual point clouds** before understanding how the 1,045 samples relate to the plants/genotypes/metadata.

The dataset has morphological and metadata information, and we need to determine whether there are repeated scans or related samples that could make a random split overly optimistic. ([Hugging Face][2])

Our target should be something conceptually like:

```text
TRAIN
70%

VALIDATION
15%

TEST
15%
```

but the actual split unit should be determined from the metadata, potentially by **plant/genotype/source group rather than individual point cloud**.

That is something we'll verify before training.

---

# 16. What about Pheno4D?

I would **not make it the primary dataset** anymore.

Pheno4D is extremely useful, but for a different experiment.

It contains seven maize and seven tomato plants scanned repeatedly over multiple days, with temporally consistent labels. ([IPB Bonn][5])

So later we could use it for:

```text
Temporal 3D phenotyping
```

rather than mixing two objectives into the main project.

For example, after Agri3D v1:

```text
Plant at t0
      ↓
Plant at t1
      ↓
Plant at t2
      ↓
growth trajectory
```

That becomes a natural **v2 research extension**.

---

# 17. The complete roadmap

So I'd break the project into these milestones:

```text
M0  Dataset + problem specification
 ↓
M1  Point-cloud ingestion
 ↓
M2  Classical 3D processing
 ↓
M3  Geometry baseline
 ↓
M4  PointNet++ segmentation
 ↓
M5  Instance extraction
 ↓
M6  Phenotypic measurements
 ↓
M7  Advanced 3D model
 ↓
M8  Robustness + benchmarking
 ↓
M9  Raw LiDAR integration
 ↓
M10 API + Docker + final report
```

And the final demonstration becomes:

```text
                 RAW DATA
                    │
          ┌─────────┴─────────┐
          │                   │
       PLY/TLS             ROS/LiDAR
          │                   │
          └─────────┬─────────┘
                    ▼
             3D PROCESSING
                    ▼
              DEEP LEARNING
                    ▼
           STRUCTURE ANALYSIS
                    ▼
             PHENOTYPING
                    ▼
          ┌─────────┴─────────┐
          ▼                   ▼
      3D VISUALIZATION     JSON/CSV
                              │
                              ▼
                           API/REPORT
```

That is the project I'd be comfortable putting on a CV/GitHub as an **end-to-end digital-agriculture 3D perception system** rather than simply a DL experiment.

### So, to lock it in

**Primary dataset:** MaizeField3D, using its **100k-point raw/unsegmented PLY clouds as model input** and its **520 annotated clouds as supervision**. ([Baskar Group][3])

**Secondary dataset:** Pullally, only for the **sensor-native LiDAR/ROS/GNSS integration stage**. ([GitHub][4])

**Optional v2:** Pheno4D for temporal plant-growth analysis. ([IPB Bonn][5])

I think this is a better scope than trying to make one dataset do everything.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC13109309/ "MaizeField3D: A curated 3D point cloud and procedural model dataset of field-grown maize from a diversity panel - PMC"
[2]: https://huggingface.co/datasets/BGLab/MaizeField3D "BGLab/MaizeField3D · Datasets at Hugging Face"
[3]: https://baskargroup.github.io/MaizeField3D/ "MaizeField3D"
[4]: https://github.com/RAL-UC/Pullally_Dataset "GitHub - RAL-UC/Pullally_Dataset: Contains the lidar scans and GNSS measurements captures at the Pullally (Chile) experimental fruit orchard. · GitHub"
[5]: https://www.ipb.uni-bonn.de/data/pheno4d/index.html "Pheno4D (2020) – StachnissLab"
